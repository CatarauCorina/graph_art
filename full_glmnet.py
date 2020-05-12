import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from glmnet_graph_learning import GraphLearning, GraphModule, loss_graph_net
from cross_graph_node_emb import CrossGraphAffinityLearning, CrossGraph
from torch_geometric.nn import GCNConv
from torch.autograd import Variable

from willow_ip import WillowDataset
from pascal_voc_ip import PascalVOCDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GLMNet(nn.Module):

    def __init__(self, size_matrix):
        super().__init__()
        self.random_adj_matrix = torch.rand(size=(size_matrix, size_matrix))
        self.random_adj_matrix = self.random_adj_matrix * (1 - torch.eye(size_matrix, size_matrix))
        self.graph_learning = GraphModule(in_features=1024, edge_attr=self.random_adj_matrix)
        self.conv1 = GCNConv(1024, 1024)
        self.conv2 = GCNConv(1024, 1024)
        #
        self.cross_graph = CrossGraph(1024)
        return

    def forward(self, x_g1, y_g2, edge_index_g1, edge_index_g2):
        x_i_1, edge_attr_1 = self.graph_learning(x_g1, edge_index_g1)
        self.x_i_proj_1, self.x_j_proj_1 = self.graph_learning.graph_learn.x_i_proj, self.graph_learning.graph_learn.x_j_proj
        y_i_2, edge_attr_2 = self.graph_learning(y_g2, edge_index_g2)
        self.y_i_proj_2, self.y_j_proj_2 = self.graph_learning.graph_learn.x_i_proj, self.graph_learning.graph_learn.x_j_proj

        e_attr_1_conv = torch.tensor([edge_attr_1[p[0], p[1]] for p in list(zip(edge_index_g1[0], edge_index_g1[1]))]).to(device)
        e_attr_2_conv = torch.tensor([edge_attr_2[p[0], p[1]] for p in list(zip(edge_index_g2[0], edge_index_g2[1]))]).to(device)
        e_attr_1_conv = Variable(e_attr_1_conv, requires_grad=True)
        e_attr_2_conv = Variable(e_attr_2_conv, requires_grad=True)
        x_i_conv_out = F.relu(self.conv1(x_i_1, edge_index_g1, e_attr_1_conv))
        y_i_conv_out = F.relu(self.conv1(y_i_2, edge_index_g2, e_attr_2_conv))
        # x_i_f = F.relu(self.conv2(x_i_conv_out, edge_index_g1, e_attr_1_conv))
        # y_i_f = F.relu(self.conv2(y_i_conv_out, edge_index_g2, e_attr_2_conv))


        x_i_cg, y_i_cg, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2 = self.cross_graph(x_i_conv_out, y_i_conv_out, edge_index_g1, edge_index_g2)

        return x_i_cg, y_i_cg, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2

