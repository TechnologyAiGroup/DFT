# coding=utf-8
import os
from data.dataloader import dataloader
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from prop_model import DeProp
from util import gradient_penalty, get_pairwise_sim, torch_corr
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
from dglutil_transformer import LapEig_positional_encoding, GTModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='dblp')
parser.add_argument("--target", type=str, default='citation')
parser.add_argument("--dataset_name", type=str, default='citation')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int, default=200)
parser.add_argument("--encoder_dim", type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--orth', type=bool, default=True)
parser.add_argument('--lambda1', type=float, default=100)
parser.add_argument('--lambda2', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.01)
parser.add_argument('--with_bn', type=bool, default=True)
parser.add_argument('--F_norm', type=bool, default=False)
parser.add_argument('--smooth', type=bool, default=True)
parser.add_argument('--lambda_gp', type=int, default=10)
parser.add_argument('--lambda_b', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--pos_enc_size', type=int, default=2)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim


id = "source: {}, target: {}, seed: {},encoder_dim: {}"\
    .format(args.source, args.target, seed, encoder_dim)

print(id)


def edge_index_to_adjacency_matrix(edge_index, num_nodes):

    adj_matrix = torch.zeros((num_nodes, num_nodes))

    for i in range(edge_index.shape[1]):
        src_node = edge_index[0, i]
        tgt_node = edge_index[1, i]
        adj_matrix[src_node, tgt_node] = 1
        adj_matrix[tgt_node, src_node] = 1  # delete when directed

    return adj_matrix



random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dataset = dataloader(task_type='nc', model_name='pyg',
                     dataset_name=args.dataset_name, collection_name=args.source)

source_data = dataset[0].to(device)
source_adj_matrix = edge_index_to_adjacency_matrix(
    source_data.edge_index, source_data.num_nodes)

source_vec_degree = source_adj_matrix.sum(dim=-1)

source_pos_enc = LapEig_positional_encoding(
    source_data.x.shape[0], source_vec_degree, source_adj_matrix, args.pos_enc_size
)
print(source_data)
dataset = dataloader(task_type='nc', model_name='pyg',
                     dataset_name=args.dataset_name, collection_name=args.target)
target_data = dataset[0].to(device)
target_adj_matrix = edge_index_to_adjacency_matrix(
    target_data.edge_index, target_data.num_nodes)
target_vec_degree = target_adj_matrix.sum(dim=-1)

target_pos_enc = LapEig_positional_encoding(
    target_data.x.shape[0], target_vec_degree, target_adj_matrix, args.pos_enc_size
)
print(target_data)

if len(dataset[0].y.shape) == 2:
    source_data.y = source_data.y.argmax(dim=-1)
    target_data.y = target_data.y.argmax(dim=-1)
source_data = source_data.to(device)
target_data = target_data.to(device)

loss_func = nn.CrossEntropyLoss().to(device)


encoder = DeProp(dataset.num_features, hidden_channels=64,
                 out_channels=encoder_dim, dropout=args.dropout, args=args,).to(device)


cls_model = nn.Sequential(
    nn.Linear(encoder_dim, encoder_dim//2),
    nn.ReLU(),
    nn.Linear(encoder_dim//2, encoder_dim//4),
    nn.ReLU(),
    nn.Linear(encoder_dim//4, dataset.num_classes)
).to(device)

domain_model = nn.Sequential(
    nn.Linear(encoder_dim, 1),
    nn.Sigmoid()
).to(device)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


Gtransformer = GTModel(out_size=encoder_dim,
                       pos_enc_size=args.pos_enc_size).cuda()
att_model = Attention(encoder_dim).cuda()

models = [encoder,  Gtransformer, cls_model]

models.extend([att_model])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, lr=args.lr)
optimizer_critic = torch.optim.Adam(domain_model.parameters(), lr=args.lr)
models.extend(domain_model)


def gcn_encode(data, adj, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, None, "gcn")
    encoder.loss_corr = 0
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = encoder(
        data.x, data.ppmi_edge_index, data.ppmi_edge_attr, "ppmi")
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    gcn_output = gcn_encode(data, cache_name, mask)
    ppmi_output = ppmi_encode(data, cache_name, mask)
    outputs = att_model([gcn_output, ppmi_output])
    if cache_name == 'source':
        return Gtransformer(
            outputs.shape[0], source_data.edge_index.cuda(), outputs, source_pos_enc.cuda())
    else:
        return Gtransformer(
            outputs.shape[0], target_data.edge_index.cuda(), outputs, target_pos_enc.cuda())


def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits, encoded_output


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    micro_f1 = f1_score(labels.detach().cpu().numpy(),
                        preds.detach().cpu().numpy(), average="micro")
    macro_f1 = f1_score(labels.detach().cpu().numpy(),
                        preds.detach().cpu().numpy(), average="macro")
    return accuracy, micro_f1, macro_f1


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits, encoded_output = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy, micro_f1, macro_f1 = evaluate(preds, labels)
    return micro_f1, macro_f1, encoded_output, accuracy


loss_list = []
source_loss_list = []
target_loss_list = []
domain_loss_list = []
epochs = 500
target_acc = []


def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()
    optimizer_critic.zero_grad()


    encoded_source = encode(source_data, "source")
    encoded_target = encode(target_data, "target")

    source_logits = cls_model(encoded_source)


    CRITIC_ITERATIONS = 10
    for inner_iter in range(CRITIC_ITERATIONS):

        source_domain_preds = domain_model(encoded_source).reshape(-1)
        target_domain_preds = domain_model(encoded_target).reshape(-1)
        gp = gradient_penalty(
            domain_model, encoded_source, encoded_target, 'gpu')
        loss_critic = (
            -torch.abs(torch.mean(source_domain_preds) -
                       torch.mean(target_domain_preds))+args.lambda_gp*gp
        )
        optimizer_critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        optimizer_critic.step()

    source_domain_preds_1 = domain_model(encoded_source).reshape(-1)
    target_domain_preds_1 = domain_model(encoded_target).reshape(-1)
    loss_grl = torch.abs(torch.mean(source_domain_preds_1) -
                         torch.mean(target_domain_preds_1))
    cls_loss = loss_func(source_logits, source_data.y)

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3
    loss = cls_loss + loss_grl*args.lambda_b

    target_logits = cls_model(encoded_target)
    target_probs = F.softmax(target_logits, dim=-1)
    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

    loss_entropy = torch.mean(
        torch.sum(-target_probs * torch.log(target_probs), dim=-1))

    loss = loss + loss_entropy * (epoch / epochs * 0.01)

    source_loss_list.append(cls_loss)
    domain_loss_list.append(loss_grl)
    target_loss_list.append(loss_entropy)
    loss_list.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
best_mif1 = 0
best_maf1 = 0

for epoch in range(1, epochs):
    train(epoch)
    src_micro_f1, src_macro_f1, src_x, source_correct = test(
        source_data, "source")
    tgt_micro_f1, tgt_macro_f1, tgt_x, target_correct = test(
        target_data, "target")
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(
        epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
        best_src_x = src_x
        best_tgt_x = tgt_x
    if tgt_micro_f1 > best_mif1:
        best_mif1 = tgt_micro_f1
    if tgt_macro_f1 > best_maf1:
        best_maf1 = tgt_macro_f1

print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}"\
    .format(id, best_epoch, best_source_acc, best_target_acc)

print(line)
print(best_mif1)
print(best_maf1)
