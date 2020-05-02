import os
import torch
import glob
import numpy as np
import torchvision
from scipy.io import loadmat
import torchvision.transforms as T
from torch.utils import data as data
from PIL import Image as PILImage
from torchvision import transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from PIL import Image
from glmnet_graph_learning import GraphLearning, GraphModule, loss_graph_net


class WillowDataset(InMemoryDataset):

    # categories = ['face', 'motorbike', 'car', 'duck', 'winebottle']
    categories = ['duck']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    def __init__(self, root='./data/willow_processed/', category='face', transform=None, pre_transform=None):
        self.root = root
        self.category = category

        super(WillowDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'willow/')

    @property
    def processed_file_names(self):
        return ['./processed']

    @property
    def raw_file_names(self):
        return [category.capitalize() for category in self.categories]

    def load_vgg_processed_inputs(self,
                                  data_loader,
                                  data_list,
                                  layers_to_use={'4_2': 20, '5_1': 25}):
        vgg16_outputs = []

        def hook(module, x, y):
            vgg16_outputs.append(y.to('cpu'))
        vgg16 = torchvision.models.vgg16(pretrained=True).to(self.device)
        features = vgg16.features
        vgg16.eval()
        for layer in layers_to_use.keys():
            layer_feature = features[layers_to_use[layer]].register_forward_hook(hook)

        for i, batch_img in enumerate(data_loader):
            vgg16_outputs.clear()

            with torch.no_grad():
                vgg16(batch_img.to(self.device))

            # Make feature maps of 256x256
            out1 = F.interpolate(vgg16_outputs[0], (256, 256), mode='bilinear',
                                 align_corners=False)
            out2 = F.interpolate(vgg16_outputs[1], (256, 256), mode='bilinear',
                                 align_corners=False)
            for j in range(out1.size(0)):
                data = data_list[i * self.batch_size + j]
                idx = data.pos.round().long().clamp(0, 255)
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                data.img = None
                data.x = torch.cat([x_1.t(), x_2.t()], dim=1)
                data.edge_attr = torch.zeros(size=(data.x.shape[0], data.x.shape[0]))
            del out1
            del out2

        return data_list

    def process(self):
        all_data = []
        for category in self.categories:
            self.path_category = os.path.join(self.raw_dir, category, '*.png')
            self.names = glob.glob(self.path_category)
            self.names = sorted([name[:-4] for name in self.names])

            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            data_list = []
            for img_name in self.names:
                x = loadmat(f'{img_name}.mat')
                point = x['pts_coord']

                x, y = torch.from_numpy(point).to(torch.float)
                pos = torch.stack([x, y], dim=1)
                with open('{}.png'.format(img_name), 'rb') as f:
                    img = Image.open(f).convert('RGB')
                pos[:, 0] = pos[:, 0] * 256.0 / (img.size[0])
                pos[:, 1] = pos[:, 1] * 256.0 / (img.size[1])

                keypoint_list = []
                for idx, keypoint in enumerate(np.split(point, point.shape[1], axis=1)):
                    attr = {'name': idx}
                    attr['x'] = float(keypoint[0]) * 256.0 / (img.size[0])
                    attr['y'] = float(keypoint[1]) * 256.0 / (img.size[1])
                    keypoint_list.append(attr)

                img = img.resize((256, 256), resample=Image.BICUBIC)
                img = transform(img)
                res = [[], []]
                for el in range(10):
                    first_lst = list(range(el + 1, 10))
                    second_list = list(np.full(len(first_lst), el))
                    res[0] = res[0] + second_list
                    res[1] = res[1] + first_lst
                edge_index = torch.tensor(res, dtype=torch.long)

                data = Data(img=img, pos=pos, name=img_name, edge_index=edge_index, keypoints=keypoint_list)
                data_list.append(data)
            imgs = [data.img for data in data_list]
            loader = DataLoader(imgs, self.batch_size, shuffle=False)
            data_list = self.load_vgg_processed_inputs(loader, data_list)
            filter_out = [data for data in data_list if data.x.shape[0] == 10]
            all_data = all_data + filter_out
        torch.save(self.collate(all_data), self.processed_paths[0])
        return


def main():
    import random
    wl_ds = WillowDataset()
    print(wl_ds[0])
    for val_x, val_y in zip(wl_ds[0::2], wl_ds[1::2]):
        x, edge_index_x, edge_attr_x = val_x.x, val_x.edge_index, val_x.edge_attr
        y, edge_index_y, edge_attr_y = val_y.x, val_y.edge_index, val_y.edge_attr
        perm_mat = np.zeros([10,10], dtype=np.float32)
        row_list = []
        col_list = []
        random.shuffle(val_x.keypoints[0])
        random.shuffle(val_y.keypoints[0])
        for i, keypoint in enumerate(val_x.keypoints[0]):
            for j, _keypoint in enumerate(val_y.keypoints[0]):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
    return


if __name__ == '__main__':
    main()





