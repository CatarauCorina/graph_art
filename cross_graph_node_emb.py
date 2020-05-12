import torch
import math
from scipy.linalg import expm
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree
from geomloss import SamplesLoss
from sinkhorn import Sinkhorn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CrossGraph(nn.Module):
    def __init__(self, dim):
        super(CrossGraph, self).__init__()  # "Add" aggregation.

        self.affinity_matrix = CrossGraphAffinityLearning(dim)
        self.gconv_1 = GCNConv(1024, 1024)
        self.sinkhorn_layer = Sinkhorn()

    def forward(self, x_g1, y_g2, edge_index_g1, edge_index_g2):
        # Step 1: Add self-loops to the adjacency matrix.
        c_xy = self.affinity_matrix(x_g1, y_g2, edge_index_g1, edge_index_g2)
        c_yx = self.affinity_matrix(y_g2, x_g1, edge_index_g2, edge_index_g1)
        c_xy_aff = c_xy.clone().unsqueeze(0)
        c_yx_aff = c_yx.clone().unsqueeze(0)

        dsm_xy = self.sinkhorn_layer(c_xy_aff).squeeze(0)
        dsm_yx = self.sinkhorn_layer(c_yx_aff).squeeze(0)

        e_attr_1_conv = torch.tensor([dsm_xy[p[0], p[1]] for p in list(zip(edge_index_g1[0], edge_index_g1[1]))]).to(device)
        e_attr_2_conv = torch.tensor([dsm_yx[p[0], p[1]] for p in list(zip(edge_index_g2[0], edge_index_g2[1]))]).to(device)

        out_conv_xy = self.gconv_1(y_g2, edge_index_g2, e_attr_1_conv)
        out_conv_yx = self.gconv_1(x_g1, edge_index_g1, e_attr_2_conv)

        #
        # edge_index_g1, _ = add_self_loops(edge_index_g1, num_nodes=x_g1.size(0))
        # edge_index_g2, _ = add_self_loops(edge_index_g2, num_nodes=y_g2.size(0))


        return out_conv_xy, out_conv_yx, edge_index_g1, edge_index_g2, dsm_xy, dsm_yx


class CrossGraphAffinityLearning(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.affinity_param = torch.nn.Parameter(Tensor(dim,dim), requires_grad=True)
        self.reset_parameters(dim)
        return

    def reset_parameters(self,d):
        stdv = 1. / math.sqrt(d)
        self.affinity_param.data.uniform_(-stdv, stdv)
        self.affinity_param.data += torch.eye(d)


    def forward(self, x_g1, x_g2, edge_index_g1, edge_index_g2, tau=0.3):
        m_ij = torch.matmul(x_g1, self.affinity_param)
        m_ij = torch.div(torch.matmul(m_ij, x_g2.T), tau)

        return m_ij


class SinkhornLayer(nn.Module):
    def __init__(self, n_iters=21, temperature=0.01, **kwargs):
        self.supports_masking = False
        self.n_iters = n_iters
        self.temperature = temperature
        super(SinkhornLayer, self).__init__(**kwargs)

    def forward(self, input_tensor, mask=None):
        n = input_tensor.shape[1]
        log_alpha = input_tensor.view([-1, n, n])
        log_alpha /= self.temperature

        for _ in range(self.n_iters):
            log_alpha -= torch.log(torch.sum(torch.exp(log_alpha), dim=2)).view([-1,n,1])
            log_alpha -= torch.log(torch.sum(torch.exp(log_alpha), dim=1)).view([-1,1,n])

        return torch.exp(log_alpha)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape

#
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#         # Step 2: Linearly transform node feature matrix.
#         x = F.relu(self.lin(x))
#
#         # Step 3: Compute normalization
#         row, col = edge_index
#         norm = row*col
#
#         # Step 4-6: Start propagating messages.
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
#                               norm=norm)
#
#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]
#
#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j
#
#     def update(self, aggr_out):
#         # aggr_out has shape [N, out_channels]
#
#         # Step 6: Return new node embeddings.
#         return aggr_out


