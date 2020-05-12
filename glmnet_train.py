import torch
import torchvision
from torch import optim
from glmnet_graph_learning import GraphLearning, GraphModule, loss_graph_net

from full_glmnet import GLMNet
import numpy as np
from pairs_dataset import PairsDS
from torch_geometric.data import DataLoader
import random
from willow_ip import WillowDataset
from cross_entropy_perm_loss import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train(counter, writer, train_dataset, optimizer, model, criterion, epoch_loss):


    for idx, data in enumerate(train_dataset):
        pair = data['pair']
        val_x = pair[0]
        val_y = pair[1]
        x, edge_index_x, edge_attr_x = val_x.x, val_x.edge_index, val_x.edge_attr
        y, edge_index_y, edge_attr_y = val_y.x, val_y.edge_index, val_y.edge_attr
        x = x.to(device)
        y = y.to(device)
        edge_attr_x = edge_attr_x.to(device)
        edge_attr_y = edge_attr_y.to(device)
        edge_index_x =edge_index_x.to(device)
        edge_index_y =edge_index_y.to(device)
        perm_matrix = torch.tensor(data['perm_matrix']).to(device)

        x_i_1, y_i_2, edge_index_g1, edge_index_g2, edge_attr_1, edge_attr_2 = model(x, y, edge_index_x, edge_index_y)

        if idx % 2 == 0:
            graph_param_1 = edge_attr_1.clone()
            loss_graph_learn = loss_graph_net(model.x_i_proj_1, model.x_j_proj_1, graph_param_1, edge_index_x)
        else:
            graph_param_2 = edge_attr_2.clone()
            loss_graph_learn = loss_graph_net(model.y_i_proj_2, model.y_j_proj_2, graph_param_2, edge_index_y)


        loss_glmnet = criterion(edge_attr_1.unsqueeze(0), perm_matrix.unsqueeze(0), data['n1_gt'], data['n2_gt'])
        old_params = {}
        params_not_updated = []
        for name, params in model.named_parameters():
            old_params[name] = params.clone()

        loss = loss_glmnet + loss_graph_learn
        loss.backward()
        print(list(old_params.keys()))
        for name, params in model.named_parameters():
            if (old_params[name] == params).all():
                params_not_updated.append(name)
            if params.is_leaf and params.grad is not None:
                print(f'{name}: {torch.sum(params.grad)}')
            elif params.is_leaf:
                print(f'{name}: {params.grad}')


        optimizer.step()

        #print(params_not_updated)
        # if len(params_not_updated) > 3:
        #     print(counter)

        optimizer.zero_grad()
        epoch_loss += loss
        writer.add_scalar('Loss/iter', loss.item(), counter)
        print(loss.item())
        counter += 1

    return epoch_loss / len(train_dataset), counter, model


def run_validation(val_dataset, model):
    validation_loss = 0
    for val in val_dataset:
        x, edge_index, edge_attr = val.x, val.edge_index, val.edge_attr
        x.to(device)
        edge_index.to(device)
        edge_attr.to(device)
        x_i, edge_attr = model(x, edge_index)
        x_i_proj, x_j_proj = model.graph_learn.x_i_proj, model.graph_learn.x_j_proj
        graph_param = edge_attr
        loss = loss_graph_net(x_i_proj, x_j_proj, graph_param, edge_index)
        validation_loss += loss
    return validation_loss / len(val_dataset)





def main():
    epochs = 100
    counter = 0
    ds_train = PairsDS()
    loader_train = DataLoader(ds_train, 1, shuffle=True)
    ds_val = PairsDS(mode='valid')
    epoch_loss = 0
    model = GLMNet(ds_train[0]['pair'][0].x.shape[0])
    criterion = CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    count_validation = 0
    not_improved = 0
    prev_validation = 100
    writer = SummaryWriter('graph/glmnet_pairs')
    for epoch in range(epochs):
        loss_epoch, counter, model = train(
            counter, writer,
            loader_train, optimizer,
            model, criterion, epoch_loss
        )

        writer.add_scalar('Loss/epoch', loss_epoch, epoch)
        # if epoch % 3:
        #     with torch.no_grad():
        #         count_validation = count_validation + 1
        #         loss_validation = run_validation(ds_val, model)
        #         writer.add_scalar('Loss validation', loss_validation, count_validation)
        #         if loss_validation >= prev_validation:
        #             not_improved += 1
        #         else:
        #             prev_validation = loss_validation
        # if not_improved == 3:
        #     torch.save(model.state_dict(), 'glmnet_test.pth')
        #     break

    return


if __name__ == '__main__':
    main()


