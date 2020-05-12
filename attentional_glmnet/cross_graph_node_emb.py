import torch
import math
from scipy.linalg import expm
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, GCNConv, SGConv
from voting_layer import Voting

from sinkhorn import simple_sinkhorn, my_sinkhorn, my_gumbel_sinkhorn
from einops import rearrange
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import math


class CrossGraph(nn.Module):
    def __init__(self, dim,
                 batch_size,
                 size_matrix,
                 use_norm=False,
                 att_cross=True,
                 multi_head=False,kernel_type='gcn'):
        super(CrossGraph, self).__init__()  # "Add" aggregation.
        self.size_matrix = size_matrix
        params = {
            'node_size': 238,
            'lin_hidden': 50,
            'out_dim': 10,
            'coord_dim': 2,
            'in_shape': 1024,
            'nr_heads': 1,
            'attention': 'additive',
            'use_norm': use_norm,
            'N': size_matrix
        }
        self.params = params
        self.multi_head = multi_head
        if not multi_head:
            self.affinity_matrix = CrossGraphAffinityLearning(dim)
        else:
            self.affinity_matrix = MultiHeadCrossGraph(params)
        if kernel_type == 'spline':
            self.gconv_1 = SGConv(1024, 1024)
        elif kernel_type == 'gcn':
            self.gconv_1 = GCNConv(1024, 1024)

        self.batch_size = batch_size
        self.att_cross = att_cross
        self.cross_att = RelationalAttentionCross(params=params)
        self.voting_layer = Voting()

        # self.sinkhorn_layer = Sinkhorn(10)

    def forward(self, x_g1, y_g2, edge_index_g1, edge_index_g2):
        # Step 1: Add self-loops to the adjacency matrix.
        #
        if self.multi_head:
            affinity_matrix = CrossGraphAffinityLearning(1024)
            c_xy = affinity_matrix(x_g1, y_g2)
            c_yx = affinity_matrix(y_g2, x_g1)
            x_g11, y_g21, c_xy, c_yx = self.affinity_matrix(x_g1, y_g2, c_xy)
            # x_g12, y_g22, c_yx = self.affinity_matrix(y_g2, x_g1)
        else:
            c_xy = self.affinity_matrix(x_g1, y_g2)
            c_yx = self.affinity_matrix(y_g2, x_g1)
            if self.att_cross:
                x_g1, y_g2, c_xy = self.cross_att(x_g1, y_g2, c_xy)

        dsm_xy = my_sinkhorn(c_xy, multi_head=self.multi_head)
        dsm_yx = my_sinkhorn(c_yx, multi_head=self.multi_head)

        e_attr_1_conv , e_attr_2_conv = self.process_edges(edge_index_g1, edge_index_g2, dsm_xy, dsm_yx, self.multi_head)

        edge_index_1 = edge_index_g1[:, :int(edge_index_g1.shape[1] / self.batch_size)].repeat(self.batch_size, 1, 1)
        edge_index_2 = edge_index_g2[:, :int(edge_index_g2.shape[1] / self.batch_size)].repeat(self.batch_size, 1, 1)


        # edge_index_g1, _ = add_self_loops(edge_index_g1, num_nodes=x_g1.size(0))
        # edge_index_g2, _ = add_self_loops(edge_index_g2, num_nodes=y_g2.size(0))
        if self.multi_head:
            out_conv_xy, out_conv_yx = self.gcn_batch_conv(
                x_g11, y_g21,
                edge_index_2, edge_index_1,
                e_attr_1_conv, e_attr_2_conv,
                self.gconv_1,
                self.multi_head
            )

        else:
            out_conv_xy, out_conv_yx = self.gcn_batch_conv(
                x_g1, y_g2,
                edge_index_2, edge_index_1,
                e_attr_1_conv, e_attr_2_conv,
                self.gconv_1,
                self.multi_head
            )
        ngt = torch.tensor([self.size_matrix]).repeat(self.batch_size,1,1)
        dsm_xy = self.voting_layer(dsm_xy, ngt)
        dsm_yx = self.voting_layer(dsm_yx, ngt)
        return out_conv_xy, out_conv_yx, edge_index_g1, edge_index_g2, dsm_xy, dsm_yx

    def process_edges(self, edge_index_g1, edge_index_g2, matrix_1, matrix_2,multi_head):
        eg1_x = edge_index_g1[:, :int(edge_index_g1.shape[1] / self.batch_size)][0]
        eg2_x = edge_index_g2[:, :int(edge_index_g2.shape[1] / self.batch_size)][0]
        eg1_y = edge_index_g1[:, :int(edge_index_g1.shape[1] / self.batch_size)][1]
        eg2_y = edge_index_g2[:, :int(edge_index_g2.shape[1] / self.batch_size)][1]

        # Intra graph-conv learning

        if multi_head:
            e_attr_1_conv = []
            e_attr_2_conv = []
            for m1, m2 in zip(matrix_1, matrix_2):
                e_attr_1_conv = e_attr_1_conv + [torch.tensor([m[p[0], p[1]] for p in list(zip(eg1_x, eg1_y))]) for m in m1]
                e_attr_2_conv = e_attr_2_conv + [torch.tensor([m[p[0], p[1]] for p in list(zip(eg2_x, eg2_y))]) for m in m2]
            e_attr_1_conv = torch.stack(e_attr_1_conv).unsqueeze(0)
            e_attr_2_conv = torch.stack(e_attr_2_conv).unsqueeze(0)
        else:
            e_attr_1_conv = torch.stack([torch.tensor([m[p[0], p[1]] for p in list(zip(eg1_x, eg1_y))]) for m in matrix_1])
            e_attr_2_conv = torch.stack([torch.tensor([m[p[0], p[1]] for p in list(zip(eg2_x, eg2_y))]) for m in matrix_2])

        return e_attr_1_conv, e_attr_2_conv

    def gcn_batch_conv(self, x_i_1, y_i_2, edge_index_1, edge_index_2, e_attr_1_conv, e_attr_2_conv, model, multi_head):
        x_i_1_all = []
        y_i_2_all = []
        for i in range(self.batch_size):
            e_attr_1 = e_attr_1_conv[i]
            e_attr_2 = e_attr_2_conv[i]
            if multi_head:
                e_attr_1 = torch.mean(e_attr_1_conv[i], dim=0)
                e_attr_2 = torch.mean(e_attr_2_conv[i], dim=0)
            x_i_1_it = F.relu(model(x_i_1[i], edge_index_1[i], e_attr_1))
            y_i_2_it = F.relu(model(y_i_2[i], edge_index_2[i], e_attr_2))
            x_i_1_all.append(x_i_1_it)
            y_i_2_all.append(y_i_2_it)

        x_i_1 = torch.stack(x_i_1_all)
        y_i_2 = torch.stack(y_i_2_all)
        return x_i_1, y_i_2


class MultiHeadCrossGraph(nn.Module):

    def __init__(self, params):
        super(MultiHeadCrossGraph, self).__init__()
        self.params = params

        self.proj_shape = (params['in_shape'], params['node_size']*params['nr_heads'])
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.norm_shape = (params['N'], params['nr_heads'], params['N']+1, params['node_size'])
        self.norm_shape_v = (params['N'], params['node_size'])
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape_v, elementwise_affine=True)

        self.additive_k_lin = nn.Linear(params['node_size'], params['N']+1)
        self.additive_q_lin = nn.Linear(params['node_size'], params['N']+1)
        self.additive_attention_lin = nn.Linear(params['N']+1, params['N']+1)

        self.linear1 = nn.Linear(params['node_size']*params['nr_heads'], params['in_shape'])

    def prepare_cross_matrices(self, b_m1, b_m2):
        cross_attention = []
        for id_batch, m1 in enumerate(b_m1):
            for idx in range(m1.shape[0]):
                cross_attention.append(torch.cat((m1[idx, :].unsqueeze(0), b_m2[id_batch]), dim=0))
        cross_attention = torch.stack(cross_attention, dim=0)
        return rearrange(cross_attention, "(b f) m h -> b f m h", b=len(b_m1))

    def squish_multi_attention_to_cross(self, A):
        all_att_xy = []
        all_att_yx = []
        for id_batch, att in enumerate(A):
            new_att_xy = rearrange(att[:, :, self.params['N']:, :self.params['N']],"c h s d -> h c (s d)", s=1)
            new_att_yx = rearrange(att[:, :, :self.params['N'], self.params['N']:].transpose(2,3), "c h s d -> h c (s d)", s=1)
            all_att_xy.append(new_att_xy)
            all_att_yx.append(new_att_yx)
        return torch.stack(all_att_xy, dim=0), torch.stack(all_att_yx, dim=0)


    def forward(self, m1, m2):
        cross_m = self.prepare_cross_matrices(m1, m2)

        K = rearrange(self.k_proj(cross_m), "b f n (head d) -> b f head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            K = self.k_norm(K)

        Q =  rearrange(self.q_proj(cross_m), "b f n (head d) -> b f head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            Q = self.q_norm(Q)

        V_1 = rearrange(self.v_proj(m1),  "b n (head d) -> b head n d", head=self.params['nr_heads'])
        V_2 = rearrange(self.v_proj(m2),  "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            V_1 = self.v_norm(V_1)
            V_2 = self.v_norm(V_2)

        if self.params['attention'] == 'dot':
            A = torch.einsum('bhfe,bhge->bhfg', Q, K)
            A = A / np.sqrt(self.params['node_size'])
            A = torch.nn.functional.softmax(A, dim=3)
        else:
            A = torch.nn.functional.elu(self.additive_q_lin(Q) + self.additive_k_lin(K))
            A = self.additive_attention_lin(A)
            A = torch.nn.functional.softmax(A, dim=3)

        # A = torch.einsum('bfe,bge->bfg', Q, K)
        # A = A / np.sqrt(self.params['node_size'])
        # A = torch.nn.functional.softmax(A, dim=2)
        with torch.no_grad():
            self.att_map = A.clone()

        A_1, A_2 = self.squish_multi_attention_to_cross(A)
        E_1 = torch.einsum('bhfc,bhcd->bhfd', A_2, V_1)
        E_2 = torch.einsum('bhfc,bhcd->bhfd', A_1, V_2)

        E_1 = rearrange(E_1, 'b head n d -> b n (head d)')
        E_2 = rearrange(E_2, 'b head n d -> b n (head d)')
        E_1 = self.linear1(E_1)
        E_2 = self.linear1(E_2)
        E_1 = torch.relu(E_1)
        E_2 = torch.relu(E_2)
        # E = rearrange(E, "b n f -> (b n) f")

        return E_1, E_2, A_1, A_2


class RelationalAttentionCross(nn.Module):

    def __init__(self, params):
        super(RelationalAttentionCross, self).__init__()
        self.params = params

        self.proj_shape = (params['in_shape'] , params['node_size']*params['nr_heads'])
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.norm_shape = (params['N'], params['node_size'])
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)

        self.additive_k_lin = nn.Linear(params['node_size'], params['N'])
        self.additive_q_lin = nn.Linear(params['node_size'], params['N'])
        self.additive_attention_lin = nn.Linear(params['N'], params['N'])

        self.linear1 = nn.Linear(params['node_size']*params['nr_heads'], params['in_shape'])

    def forward(self, m1, m2, comb):

        K = rearrange(self.k_proj(m1), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            K = self.k_norm(K)

        Q = rearrange(self.q_proj(m2), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            Q = self.q_norm(Q)

        V_1 = rearrange(self.v_proj(m1), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        V_2 = rearrange(self.v_proj(m2), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            V_1 = self.v_norm(V_1)
            V_2 = self.v_norm(V_2)

        if self.params['attention'] == 'dot':
            A = torch.einsum('bhfe,bhge->bhfg', Q, K)
            A = A / np.sqrt(self.params['node_size'])
            A = torch.nn.functional.softmax(A, dim=3)
        else:
            A = torch.nn.functional.elu(self.additive_q_lin(Q) + self.additive_k_lin(K))
            A = self.additive_attention_lin(A)
            A = torch.nn.functional.softmax(A, dim=3)

        # A = torch.einsum('bfe,bge->bfg', Q, K)
        # A = A / np.sqrt(self.params['node_size'])
        # A = torch.nn.functional.softmax(A, dim=2)
        with torch.no_grad():
            self.att_map = A.clone()


        E_1 = torch.einsum('bhfc,bhcd->bhfd', A, V_1)
        E_2 = torch.einsum('bhfc,bhcd->bhfd', A, V_2)

        E_1 = rearrange(E_1, 'b head n d -> b n (head d)')
        E_2 = rearrange(E_2, 'b head n d -> b n (head d)')
        E_1 = self.linear1(E_1)
        E_2 = self.linear1(E_2)
        E_1 = torch.relu(E_1)
        E_2 = torch.relu(E_2)
        # E = rearrange(E, "b n f -> (b n) f")

        return E_1, E_2, A


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


    def forward(self, x_g1, x_g2, tau=0.3):
        m_ij = torch.matmul(x_g1, (self.affinity_param + self.affinity_param.transpose(0,1))/2)
        m_ij = torch.matmul(m_ij, x_g2.transpose(1,2))
        res = torch.nn.functional.softplus(m_ij) - 0.5

        return res


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


