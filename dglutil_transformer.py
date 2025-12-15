import random
from tqdm import tqdm
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from dgl.dataloading import GraphDataLoader
from dgl.data import AsGraphPredDataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import dgl.sparse as dglsp
import dgl.nn as dglnn
import dgl
import os
import torch
# os.environ['TORCH'] = torch.__version__
# os.environ['DGLBACKEND'] = "pytorch"
# try:
#     import dgl
#     installed = True
# except ImportError:
#     installed = False
# print("DGL installed!" if installed else "Failed to install DGL!")


def LapEig_positional_encoding(N, vec_degree, Adj, pos_enc_dim):
    # Adj = g.adj().to_dense() # Adjacency matrix
    # Inverse and sqrt of degree matrix
    Dn = (vec_degree.clip(1) ** -0.5).diag()
    Lap = torch.eye(N) - Dn.matmul(Adj).matmul(Dn)  # Laplacian operator
    EigVal, EigVec = torch.linalg.eig(Lap)  # Compute full EVD
    EigVal, EigVec = EigVal.real, EigVec.real  # make eig real
    # sort in increasing order of eigenvalues
    EigVec = EigVec[:, EigVal.argsort()]
    # select the first non-trivial "pos_enc_dim" eigenvector
    EigVec = EigVec[:, 1:pos_enc_dim+1]
    return EigVec


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)


class GTModel(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_size=128,
        pos_enc_size=2,
        num_layers=4,
        num_heads=1,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_size)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.pooler = dglnn.SumPooling()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
        )

    def forward(self, N, indices, X, pos_enc):
        # indices = torch.stack(g.edges())
        # N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        # h = self.atom_encoder(X) + self.pos_linear(pos_enc)
        h = X+self.pos_linear(pos_enc)
        # h=X
        for layer in self.layers:
            h = layer(A, h)
        # h = self.pooler(g, h)

        # return self.predictor(h)
        return h


@torch.no_grad()
def evaluate(model, dataloader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    for batched_g, labels in dataloader:
        batched_g, labels = batched_g.to(device), labels.to(device)
        y_hat = model(
            batched_g, batched_g.ndata["feat"], batched_g.ndata["PE"])
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["rocauc"]


def train(model, dataset, evaluator, device):
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=256,
        shuffle=True,
        collate_fn=collate_dgl,
    )
    valid_dataloader = GraphDataLoader(
        dataset[dataset.val_idx], batch_size=256, collate_fn=collate_dgl
    )
    test_dataloader = GraphDataLoader(
        dataset[dataset.test_idx], batch_size=256, collate_fn=collate_dgl
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=num_epochs, gamma=0.5
    )
    loss_fcn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batched_g, labels in train_dataloader:
            batched_g, labels = batched_g.to(device), labels.to(device)
            logits = model(
                batched_g, batched_g.ndata["feat"], batched_g.ndata["PE"]
            )
            loss = loss_fcn(logits, labels.float())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        val_metric = evaluate(model, valid_dataloader, evaluator, device)
        test_metric = evaluate(model, test_dataloader, evaluator, device)
        print(
            f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, "
            f"Val: {val_metric:.4f}, Test: {test_metric:.4f}"
        )


# Training device.
# dev = torch.device("cpu")
# # Uncomment the code below to train on GPU. Be sure to install DGL with CUDA support.
# # dev = torch.device("cuda:0")

# # Load dataset.
# pos_enc_size = 8
# dataset = AsGraphPredDataset(
#     DglGraphPropPredDataset("ogbg-molhiv", "./data/OGB")
# )
# evaluator = Evaluator("ogbg-molhiv")

# # Down sample the dataset to make the tutorial run faster.
# random.seed(42)
# train_size = len(dataset.train_idx)
# val_size = len(dataset.val_idx)
# test_size = len(dataset.test_idx)
# dataset.train_idx = dataset.train_idx[
#     torch.LongTensor(random.sample(range(train_size), 2000))
# ]
# dataset.val_idx = dataset.val_idx[
#     torch.LongTensor(random.sample(range(val_size), 1000))
# ]
# dataset.test_idx = dataset.test_idx[
#     torch.LongTensor(random.sample(range(test_size), 1000))
# ]

# # Laplacian positional encoding.
# indices = torch.cat([dataset.train_idx, dataset.val_idx, dataset.test_idx])
# for idx in tqdm(indices, desc="Computing Laplacian PE"):
#     g, _ = dataset[idx]
#     g.ndata["PE"] = dgl.laplacian_pe(g, k=pos_enc_size, padding=True)

# # Create model.
# out_size = dataset.num_tasks
# model = GTModel(out_size=out_size, pos_enc_size=pos_enc_size).to(dev)

# # Kick off training.
# train(model, dataset, evaluator, dev)
