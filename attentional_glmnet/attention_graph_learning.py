import torch
import numpy as np
import torch.nn.functional as F
from willow_ip import WillowDataset
import torch.nn as nn
from einops import rearrange
from torch import optim

from torch.utils.data import DataLoader
from pairs_dataset import PairsDS
import matplotlib.pyplot as plt
from attention_graph_match import GraphMatchAtt
from cross_entropy_perm_loss import CrossEntropyLoss
from evaluation_matching_metrics import matching_accuracy, get_pos_neg

from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib.pyplot import Line2D

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.0010, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()




def train(counter, writer, train_dataset, optimizer, model, criterion, epoch_loss, epoch_acc, epoch):

    for idx,data in enumerate(train_dataset):

        pair = data['pair']
        val_x = pair[0]
        val_y = pair[1]
        x, edge_index_x, edge_attr_x = val_x.x, val_x.edge_index, val_x.edge_attr
        y, edge_index_y, edge_attr_y = val_y.x, val_y.edge_index, val_y.edge_attr
        x = x.to(device)
        y = y.to(device)
        edge_attr_x = edge_attr_x.to(device)
        edge_attr_y = edge_attr_y.to(device)
        edge_index_x = edge_index_x.to(device)
        edge_index_y = edge_index_y.to(device)
        if len(torch.tensor(data['perm_matrix']).shape) > 2:
            perm_matrix = torch.stack([torch.stack(t) for t in data['perm_matrix']]).to(device)
        else:
            perm_matrix = torch.tensor(data['perm_matrix']).unsqueeze(0)
        val_x.pos =val_x.pos.to(device)
        val_y.pos = val_y.pos.to(device)

        x_i_1, y_i_2, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2 = model(
            x, y,
            edge_index_x, edge_index_y,
            val_x.pos, val_y.pos
        )


        loss = criterion(edge_attr_1, perm_matrix, torch.tensor(data['n1_gt']).unsqueeze(0), torch.tensor(data['n2_gt']).unsqueeze(0))

        print(loss)
        acc, total_pred_num = matching_accuracy(edge_attr_1, perm_matrix,torch.tensor(data['n1_gt']).unsqueeze(0), torch.tensor(data['n2_gt']).unsqueeze(0))
        # #tp, fp, fn = get_pos_neg(edge_attr_1, perm_matrix)
        print(acc)
        writer.add_scalar('Iter/acc', acc, counter)

        loss.backward()
        # if counter % 500 == 0:
        #     plot_grad_flow(model.named_parameters())
        print([(m[0],m[1].grad.min().item(),m[1].grad.max().item()) for m in list(model.named_parameters()) if m[1].grad is not None])
        for n, p in model.named_parameters():
            if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
                stddev = 1 / ((1 + epoch) ** 0.55)
                p.grad = torch.add(p.grad,torch.empty(p.grad.shape).normal_(mean=0,std=stddev))

        optimizer.step()

        optimizer.zero_grad()
        epoch_loss += loss
        epoch_acc +=acc
        writer.add_scalar('Iter/loss', loss.item(), counter)
        counter += 1

    return epoch_loss / len(train_dataset), epoch_acc/ len(train_dataset), counter, model

def main():
    epochs = 100
    counter = 0
    ds_to_run = 'facial'
    ds_train = PairsDS(ds_type=ds_to_run)
    # loader_train = DataLoader(ds_train, 2, shuffle=True)
    batch_size = 1
    ds_val = PairsDS(mode='valid')
    epoch_loss = 0
    epoch_acc = 0
    model = GraphMatchAtt(ds_train[0]['pair'][0].x.shape[0], batch_size,kernel_type='spline').to(device)
    criterion = CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    count_validation = 0
    not_improved = 0
    prev_validation = 100
    writer = SummaryWriter(f'graph_att/glmnet_pairs_{ds_to_run}_sconv_norm_voting')
    for idx, epoch in enumerate(range(epochs)):
        print(f'Epoch {idx}')
        epoch_loss, epoch_acc, counter, model = train(
            counter, writer,
            ds_train, optimizer,
            model, criterion, epoch_loss,
            epoch_acc,
            idx
        )
        writer.add_scalar('Epoch/loss', epoch_loss, idx)
        writer.add_scalar('Epoch/acc', epoch_acc, counter)
        plot_grad_flow(model.named_parameters())

    torch.save(model.state_dict(), f'glmnet_pairs_{ds_to_run}_sconv_norm_voting.pth')

    return


if __name__ == '__main__':
    main()