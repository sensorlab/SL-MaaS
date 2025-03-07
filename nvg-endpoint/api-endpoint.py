from typing import List
from fastapi import FastAPI
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, CrossEntropyLoss
import numpy as np
from ts2vg import NaturalVG

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, ChebConv, global_sort_pool


class GINE(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h):
        super(GINE, self).__init__()
        edge_dim = 1
        train_eps = True
        self.conv1 = GINEConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim, train_eps=train_eps)

        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim, train_eps=train_eps)
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim, train_eps=train_eps)

        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim, train_eps=train_eps)
        self.conv5 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim, train_eps=train_eps)

        self.lin1 = Linear(dim_h * 5, dim_h * 4)
        self.lin2 = Linear(dim_h * 4, 5)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr=edge_weight)
        h2 = self.conv2(h1, edge_index, edge_attr=edge_weight)
        h3 = self.conv3(h2, edge_index, edge_attr=edge_weight)
        h4 = self.conv4(h3, edge_index, edge_attr=edge_weight)
        h5 = self.conv5(h4, edge_index, edge_attr=edge_weight)

        # Graph-level readout

        h1 = global_max_pool(h1, batch)
        h2 = global_max_pool(h2, batch)
        h3 = global_max_pool(h3, batch)
        h4 = global_max_pool(h4, batch)
        h5 = global_max_pool(h5, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)  # h5,
        # h = torch.cat((h1, h2), dim=1)
        # h = h1
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h


def adjToEdgidx(adj_mat):
    edge_index = torch.from_numpy(adj_mat).nonzero().t().contiguous()
    row, col = edge_index
    edge_weight = adj_mat[row, col]  # adj_mat[row, col]
    return edge_index, edge_weight


app = FastAPI()

# Load the model at server start
model = GINE(32).double()
model.load_state_dict(torch.load("model_NVG"))
model = torch_geometric.compile(model)
model.eval()


@app.post("/NVG")
async def echo(data: List[List[float]]):
    # print(data[:5])
    data = np.array(data).reshape(300)
    g = NaturalVG(weighted="distance")
    g.build(data)
    adj_mat = g.adjacency_matrix(use_weights=True, no_weight_value=0)
    edge_index, edge_weight = adjToEdgidx(adj_mat)
    inp = Data(x=torch.unsqueeze(torch.tensor(data, dtype=torch.double), 1), edge_index=edge_index,
               edge_attr=torch.unsqueeze(torch.tensor(edge_weight, dtype=torch.double), 1))
    # print(inp)
    out = model(inp)
    pred = out.argmax(dim=1)
    # print(pred)

    return {"pred": str(pred)}
