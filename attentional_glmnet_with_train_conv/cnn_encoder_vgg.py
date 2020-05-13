import torch
import torch.nn as nn
from torchvision import models
from utils.feature_align import feature_align


class VGG16Encoder(nn.Module):
    def __init__(self, batch_norm=True, use_edge=True):
        super(VGG16Encoder, self).__init__()
        self.node_layers, self.edge_layers = self.get_backbone(batch_norm, use_edge)

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm, use_edge):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        if batch_norm:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)

        conv_layers = nn.Sequential(*list(model.features.children()))

        conv_list = node_list = edge_list = []
        edge_layers = None

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU) and use_edge:
                edge_list = conv_list
                break

        assert len(node_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        if use_edge:
            edge_layers = nn.Sequential(*edge_list)

        return node_layers, edge_layers


class VGG16_bn(VGG16Encoder):
    def __init__(self):
        super(VGG16_bn, self).__init__(True)


class VGG16(VGG16Encoder):
    def __init__(self):
        super(VGG16, self).__init__(False)


class NoBackbone(nn.Module):
    def __init__(self, batch_norm=True):
        super(NoBackbone, self).__init__()
        self.node_layers, self.edge_layers = None, None

    def forward(self, *input):
        raise NotImplementedError

class NodeEncoder(VGG16_bn):

    def __init__(self):
        super(NodeEncoder, self).__init__()
        self.l2norm = nn.LocalResponseNorm(512 * 2, alpha=512 * 2, beta=0.5, k=0)

    def forward(self, src, tgt, pos_src, pos_tgt, ns_src, ns_tgt, type='img'):
        if type == 'img' or type == 'image':
            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, pos_src, ns_src,  (256, 256))
            F_src = feature_align(src_edge, pos_src, ns_src,  (256, 256))
            U_tgt = feature_align(tgt_node, pos_tgt, ns_tgt,  (256, 256))
            F_tgt = feature_align(tgt_edge, pos_tgt, ns_tgt,  (256, 256))

        # adjacency matrices
        # A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        # A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)

        return emb1, emb2
