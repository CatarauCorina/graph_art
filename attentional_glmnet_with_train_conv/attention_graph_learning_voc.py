import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torch import optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from utils.hungarian import hungarian
from attention_graph_match import GraphMatchAtt
from cross_entropy_perm_loss import CrossEntropyLoss

from utils.evaluation_metric import matching_accuracy, matching_accuracy_voc
from data.data_loader import GMDataset, get_dataloader

from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib.pyplot import Line2D

def plot_grad_flow(named_parameters, writer):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    fig, ax = plt.subplots()
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
    fig.tight_layout()
    writer.add_figure(tag=f'gradient_flow_custom_attention',
                      figure=fig)


def train(
        counter, writer, dataloader,
        optimizer, scheduler,
        model, criterion, epoch_loss, epoch_acc, epoch):
    dataset_size = len(dataloader['train'].dataset)
    print(f'Data size {dataset_size}')

    epoch_loss = 0.0
    running_loss = 0.0

    for inputs in dataloader['train']:
        if 'images' in inputs:
            data1, data2 = [_.to(device) for _ in inputs['images']]
            inp_type = 'img'
        elif 'features' in inputs:
            data1, data2 = [_.to(device) for _ in inputs['features']]
            inp_type = 'feat'
        else:
            raise ValueError('no images found')

        pos_1, pos_2 = [_.to(device) for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.to(device) for _ in inputs['ns']]
        G1_gt, G2_gt = [_.to(device) for _ in inputs['Gs']]
        H1_gt, H2_gt = [_.to(device) for _ in inputs['Hs']]
        perm_matrix = inputs['gt_perm_mat'].to(device)

        pos_1_exp = torch.zeros((pos_1.shape[0], 19, 2))
        pos_2_exp = torch.zeros((pos_1.shape[0], 19, 2))
        pos_1_exp[:, :pos_1.shape[1], :] = pos_1
        pos_2_exp[:, :pos_2.shape[1], :] = pos_2
        pos_1_exp = pos_1_exp.to(device)
        pos_2_exp = pos_2_exp.to(device)


        x_i_1, y_i_2, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2 = \
            model(data1, data2, pos_1_exp, pos_2_exp, n1_gt, n2_gt, inp_type)

        try:
            loss = criterion(edge_attr_1, perm_matrix, n1_gt, n2_gt)
            acc, _, __ = matching_accuracy_voc(hungarian(edge_attr_1,n1_gt, n2_gt), perm_matrix,n1_gt)
        except:
            print('Err')
            print(hungarian(edge_attr_1, n1_gt, n2_gt))

        writer.add_scalar('Iter/acc', acc, counter)

        loss.backward()
        if counter % 1000 == 0:
            plot_grad_flow(model.named_parameters(),writer)
        epoch_loss += loss
        epoch_acc += acc

        if counter % 500 == 0:
            print(epoch_loss/(counter+1))
            print(epoch_acc/ (counter+1))
            writer.add_scalar('Avg/acc', epoch_acc/(counter+1), counter)
            writer.add_scalar('Avg/loss', epoch_loss / (counter + 1), counter)
            print([(m[0], m[1].grad.min().item(), m[1].grad.max().item()) for m in list(model.named_parameters()) if
                   m[1].grad is not None])

        optimizer.step()

        optimizer.zero_grad()
        writer.add_scalar('Iter/loss', loss.item(), counter)
        counter += 1

    return epoch_loss / counter, epoch_acc/ counter, counter, model

def main():
    epochs = 10
    counter = 0
    ds_to_run = 'voc'
    torch.manual_seed(123)
    batch_size = 1
    dataset_len = {
        'train': 2000 * batch_size,
        'test': 1000
    }
    image_dataset = {
        x: GMDataset(
            'PascalVOC',
            sets=x,
            length=dataset_len[x],
            cls='none' if x == 'train' else None,
            obj_resize=(256, 256)
        )
        for x in ('train', 'test')}
    if ds_to_run == 'willow':
        size_matrix = 10
    else:
        size_matrix = 19
    model = GraphMatchAtt(size_matrix, batch_size,kernel_type='spline')

    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
                  for x in ('train', 'test')}
    optimizer = optim.SGD(model.parameters(), lr=1.0e-3, momentum=0.9, nesterov=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    epoch_loss = 0
    epoch_acc = 0
    criterion = CrossEntropyLoss()
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    count_validation = 0
    not_improved = 0
    prev_validation = 100
    writer = SummaryWriter(f'graph/glmnet_pairs_{ds_to_run}_sgd_full_conv_03_att')
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[10],
                                                   gamma=0.1,
                                                   last_epoch=0 - 1)
        epoch_loss, epoch_acc, counter, model = train(
            counter, writer,
            dataloader, optimizer,scheduler,
            model, criterion, epoch_loss,
            epoch_acc,
            epoch
        )
        torch.save(model.state_dict(), f'glmnet_pairs_{ds_to_run}_sgd_full_conv_03_att_{epoch}.pth')

        writer.add_scalar('Epoch/loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/acc', epoch_acc, epoch)
        scheduler.step()
        plot_grad_flow(model.named_parameters())

    torch.save(model.state_dict(), f'glmnet_pairs_{ds_to_run}_sgd_full_conv_03_att.pth')

    return


if __name__ == '__main__':
    main()