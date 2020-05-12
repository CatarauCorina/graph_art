import torch
import numpy as np
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, n_head,concat=True,layer_id=0):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features*n_head
        self.layer_id = layer_id

        self.concat = concat
        self.n_head=n_head


        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.relu = nn.ReLU()

    def forward(self, emb1,n_src,ns_tgt):
        h = self.W(emb1)
        batch_size = h.size()[0]

        a1v = torch.matmul(h,self.a1)
        a2v = torch.matmul(h,self.a2)
        a2vT = a2v.transpose(1,2)
        e = (a1v+a2vT)/self.n_head

        max_rows = torch.max(n_src)
        max_cols = torch.max(ns_tgt)
        mask = torch.zeros(batch_size, max_rows, max_cols, device=e.device)

        for b in range(batch_size):
            b_rows = slice(0, n_src[b])
            b_cols = slice(0, ns_tgt[b])
            mask[b, b_rows, b_cols] = 1

        zero_vec = torch.zeros_like(e)
        attention = self.relu(e)
        attention = torch.where(mask > 0, attention, zero_vec)

        topk_attention=torch.zeros_like(attention)
        base_num=torch.ones((1,))*2
        base_ratio=torch.ones((1))*0.1
        scale_value=torch.zeros((batch_size,1,1),device=attention.device)
        for b in range(batch_size):
            battention=attention[b]
            ratio=torch.max(0.4/np.power(base_num,self.layer_id),base_ratio)
            link_num = n_src[b] * ratio[0]
            scale_value[b,0,0]=1/link_num
            valuem, index = torch.topk(battention, link_num.long(), dim=-1)

            btopk_attention=torch.zeros_like(battention)
            btopk_attention.scatter_(1, index, valuem)

            ones_t = torch.ones_like(btopk_attention)
            zeros_t = torch.zeros_like(btopk_attention)
            label_m = torch.where(btopk_attention > 0, ones_t, zeros_t)
            topk_attention[b] = btopk_attention * label_m.transpose(0, 1)

        attention = scale_value*torch.tanh(topk_attention)
        return attention
