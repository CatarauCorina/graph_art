import torch
import torchvision
from torch import optim
from glmnet_graph_learning import GraphLearning, GraphModule, loss_graph_net
from willow_ip import WillowDataset
from pascal_voc_ip import PascalVOCDataset
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(counter, writer, train_dataset, optimizer, model, epoch_loss):
    print(counter)
    print(len(train_dataset))
    for val in train_dataset:
        x, edge_index, edge_attr = val.x, val.edge_index, val.edge_attr
        x = x.to(device)
        edge_attr = edge_attr.to(device)
        edge_index = edge_index.to(device)
        x_i, edge_attr = model(x, edge_index)
        #print(counter)

        x_i_proj, x_j_proj = model.graph_learn.x_i_proj, model.graph_learn.x_j_proj
        graph_param = edge_attr.clone()

        loss = loss_graph_net(x_i_proj, x_j_proj, graph_param, edge_index)

        writer.add_scalar('Loss/iter', loss, counter)
        loss.backward()
        for name, params in model.named_parameters():
            if params.is_leaf and params.grad is not None:
                print(f'{name} {params.is_leaf}: {torch.sum(params.grad)}')
            elif params.grad is None:
                print(f'{name} {params.is_leaf}: {params.grad}')
            else:
                print(f't {name} {params.is_leaf}')
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss
        counter += 1
    return epoch_loss / len(train_dataset), counter, model


def run_validation(val_dataset, model):
    validation_loss = 0
    for val in val_dataset:
        x, edge_index, edge_attr = val.x, val.edge_index, val.edge_attr
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        x_i, edge_attr = model(x, edge_index)
        x_i_proj, x_j_proj = model.graph_learn.x_i_proj, model.graph_learn.x_j_proj
        graph_param = edge_attr
        loss = loss_graph_net(x_i_proj, x_j_proj, graph_param, edge_index)
        validation_loss += loss
    return validation_loss / len(val_dataset)


def main():
    epochs = 30
    counter = 0
    dataset = 'willow'
    if dataset == 'voc':
        ds = PascalVOCDataset()
        size_matrix = 23
        size_features = size_matrix*size_matrix
    if dataset == 'willow':
        ds = WillowDataset()
        size_matrix = 10
        size_features = size_matrix * size_matrix
    dataset = ds.shuffle()
    print(len(dataset))
    train_dataset = dataset[:(len(dataset) - 10)]
    val_dataset = dataset[(len(dataset) - 10):len(dataset)]
    epoch_loss = 0
    random_adj_matrix = torch.rand(size=(size_matrix, size_matrix))
    random_adj_matrix = random_adj_matrix * (1 - torch.eye(size_matrix, size_matrix))
    model = GraphModule(in_features=1024, edge_attr=random_adj_matrix)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    count_validation = 0
    not_improved = 0
    prev_validation = 100
    writer = SummaryWriter('graph_new/voc_graph_learning2')
    for epoch in range(epochs):
        loss_epoch, counter, model = train(
            counter, writer,
            train_dataset, optimizer,
            model, epoch_loss
        )
        print(epoch)
        writer.add_scalar('Loss/epoch', loss_epoch, epoch)
        if epoch % 10 == 0:
            with torch.no_grad():
                count_validation = count_validation + 1
                loss_validation = run_validation(val_dataset, model)
                writer.add_scalar('Loss/validation', loss_validation, epoch)
                if loss_validation >= prev_validation:
                    not_improved += 1
                else:
                    prev_validation = loss_validation
        if not_improved == 3:
            torch.save(model.state_dict(), 'person_voc_gl.pth')
            break
    torch.save(model.state_dict(),'person_graph_lr_all.pth')


    return


if __name__ == '__main__':
    main()


