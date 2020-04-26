import torch
from numpy import linalg
import torch_geometric.nn as geo_nn
from torch_geometric.utils import scatter_
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import softmax
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn import GATConv, AGNNConv
from torch_geometric.datasets import PascalVOCKeypoints, WILLOWObjectClass


class GraphModule(nn.Module):

    def __init__(self, in_features, edge_attr):
        super().__init__()
        self.graph_learn = GraphLearning(in_features)
        self.edge_values = Variable(edge_attr, requires_grad=True)
        return

    def forward(self, x, edge_index):
        x_i, self.edge_values = self.graph_learn(x, edge_index, self.edge_values)
        return x_i, self.edge_values


class GraphLearning(geo_nn.MessagePassing):

    def __init__(self, in_features):
        super(GraphLearning, self).__init__(aggr='max')
        self.projection_embedding = nn.Linear(in_features=in_features, out_features=200)
        self.relu_msg = nn.ReLU()
        self.attention = nn.Linear(in_features=200, out_features=1)

        return

    def message(self, x, x_i, x_j, edge_index, edge_attr):
        edge_values = edge_attr.clone()
        x_i_projected = self.projection_embedding(x_i)
        x_j_projected = self.projection_embedding(x_j)
        self.x_i_proj = x_i_projected
        self.x_j_proj = x_j_projected
        diff = torch.abs(x_i_projected - x_j_projected)
        alpha = self.attention(diff)
        alpha = self.relu_msg(alpha)
        alpha = softmax(alpha, dim=0)
        for idx, idx_edge in enumerate(edge_index.T):
            edge_values[tuple(idx_edge)] = alpha[idx]
            edge_values[(idx_edge[1], idx_edge[0])] = alpha[idx]
        return x_i, edge_values

    def aggregate(self, inputs, index, dim_size):
        x_i, edge_attr = inputs
        return scatter_(self.aggr, x_i, index, self.node_dim, dim_size), edge_attr

    def update(self, aggr_out):
        return aggr_out

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = remove_self_loops(edge_index)
        x_i, edge_attr_res = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return x_i, edge_attr_res


def loss_graph_net(x_i, x_j, edge_attr, edge_index, gamma=0.1):
    euclidean_distance_nodes = torch.pairwise_distance(x_i, x_j, 2)
    loss_sum = 0
    for idx, edge_index in enumerate(edge_index.T):
        loss_sum = loss_sum + euclidean_distance_nodes[idx]*edge_attr[tuple(edge_index)]
    final_loss = loss_sum + gamma*torch.norm(edge_attr, p='fro')
    return final_loss
