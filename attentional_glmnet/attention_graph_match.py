import torch
from cross_graph_node_emb import CrossGraphAffinityLearning, CrossGraph
from torch_geometric.nn import GCNConv, SGConv

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from einops import rearrange
import numpy as np


class RelationalAttention(nn.Module):

    def __init__(self, params):
        super(RelationalAttention, self).__init__()
        self.params = params

        self.proj_shape = (params['in_shape'] + 2, params['node_size']*params['nr_heads'])
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

    def forward(self, x, pos):
        pos = torch.tensor(pos, dtype=torch.float32)
        x = rearrange(x, "(b n) f -> b n f", n=self.params['N'])
        pos = rearrange(pos, "(b n) f -> b n f", n=self.params['N'])

        x = torch.cat([x, pos], dim=2)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            K = self.k_norm(K)

        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            Q = self.q_norm(Q)

        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.params['nr_heads'])
        if self.params['use_norm']:
            V = self.v_norm(V)

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

        E = torch.einsum('bhfc,bhcd->bhfd', A, V)
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        # E = rearrange(E, "b n f -> (b n) f")

        return E, A


class GraphMatchAtt(nn.Module):

    def __init__(self, size_matrix, batch_size,
                 multi_head=True,
                 use_norm=True,
                 kernel_type='gcn'
                 ):
        super().__init__()
        random_adj_matrix = torch.rand(size=(size_matrix, size_matrix))
        random_adj_matrix = random_adj_matrix * (1 - torch.eye(size_matrix, size_matrix))
        params = {
            'node_size': 128,
            'lin_hidden': 50,
            'out_dim': 10,
            'coord_dim': 2,
            'in_shape': 1024,
            'nr_heads': 3,
            'attention': 'additive',
            'use_norm': use_norm,
            'N': size_matrix
        }
        self.batch_size = batch_size
        self.graph_learning = RelationalAttention(params)
        if kernel_type == 'gcn':
            self.conv1 = GCNConv(1024, 1024)
            self.conv2 = GCNConv(1024, 1024)

        elif kernel_type == 'spline':
            self.conv1 = SGConv(1024, 1024)
            self.conv2 = SGConv(1024, 1024)


        self.multi_head = multi_head
        self.cross_graph = CrossGraph(
            1024, batch_size, size_matrix=size_matrix, use_norm=True,
            multi_head=False, kernel_type=kernel_type, att_cross=False)
        return

    def forward(self, x_g1, y_g2, edge_index_g1, edge_index_g2, pos_g1, pos_g2):
        #Graph attention learning
        x_i_1, edge_attr_1 = self.graph_learning(x_g1, pos_g1)
        y_i_2, edge_attr_2 = self.graph_learning(y_g2, pos_g2)
        with torch.no_grad():
            self.att_1 = edge_attr_1
            self.att_2 = edge_attr_2

        #Intra graph-conv learning
        e_attr_1_conv , e_attr_2_conv = self.cross_graph.process_edges(
            edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2, self.multi_head
        )
        edge_index_1 = edge_index_g1[:, :int(edge_index_g1.shape[1]/self.batch_size)].repeat(self.batch_size,1,1)
        edge_index_2 = edge_index_g2[:, :int(edge_index_g2.shape[1]/self.batch_size)].repeat(self.batch_size,1,1)

        x_i_1, y_i_2 = self.cross_graph.gcn_batch_conv(
            x_i_1, y_i_2, edge_index_1, edge_index_2, e_attr_1_conv, e_attr_2_conv, self.conv1,self.multi_head
        )

        x_i_1, y_i_2 = self.cross_graph.gcn_batch_conv(
            x_i_1, y_i_2, edge_index_1, edge_index_2, e_attr_1_conv, e_attr_2_conv, self.conv2, self.multi_head
        )
        x_i_1, y_i_2, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2 = self.cross_graph(
            x_i_1, y_i_2,
            edge_index_g1,
            edge_index_g2
        )


        return x_i_1, y_i_2, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2

