import os
import torch
import glob
import cv2
import numpy as np
import torchvision
import pandas as pd
import torchvision.transforms as T

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from PIL import Image


class FacialDataset(InMemoryDataset):
    # categories = ['face', 'motorbike', 'car', 'duck', 'winebottle']
    categories = ['face']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    def __init__(self, root='../data/facial_processed/', category='face', transform=None, pre_transform=None):
        self.root = root
        self.category = category

        super(FacialDataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'facial/')

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
        face_images_db = np.load(os.path.join(self.raw_dir, 'face_images.npz'))['face_images']
        facial_keypoints_df = pd.read_csv(os.path.join(self.raw_dir, 'facial_keypoints.csv'))
        numMissingKeypoints = facial_keypoints_df.isnull().sum(axis=1)
        allKeypointsPresentInds = np.nonzero(numMissingKeypoints == 0)[0]

        faceImagesDB = face_images_db[:, :, allKeypointsPresentInds]
        facialKeypointsDF = facial_keypoints_df.iloc[allKeypointsPresentInds, :].reset_index(drop=True)

        (imHeight, imWidth, numImages) = faceImagesDB.shape
        numKeypoints = facialKeypointsDF.shape[1] / 2

        img_tensor = torch.tensor(faceImagesDB)
        img_tensor = img_tensor.permute(2, 0, 1)

        transform = T.Compose([
            T.ToTensor(),
        ])

        data_list = []
        for idx, img_arr in enumerate(img_tensor):
            x = torch.tensor(np.array(facialKeypointsDF.iloc[idx, 0:30:2].tolist()))
            y = torch.tensor(np.array(facialKeypointsDF.iloc[idx, 1:30:2].tolist()))

            pos = torch.stack([x, y], dim=1)
            pos[:, 0] = pos[:, 0] * 256.0 / (img_arr.shape[0])
            pos[:, 1] = pos[:, 1] * 256.0 / (img_arr.shape[1])

            keypoint_list = []
            for idx, keypoint in enumerate(list(zip(x, y))):
                attr = {'name': idx}
                attr['x'] = float(keypoint[0]) * 256.0 / (img_arr.shape[0])
                attr['y'] = float(keypoint[1]) * 256.0 / (img_arr.shape[1])
                keypoint_list.append(attr)
            img = Image.fromarray(img_arr.numpy())
            image_rgb = torch.tensor(img_arr.numpy(), dtype=torch.float32)
            image_rgb = image_rgb.view(96, 96, 1).expand(-1, -1, 3)

            img = img.resize((256, 256), resample=Image.BICUBIC)
            img = transform(img)
            res = [[], []]
            for el in range(len(x)):
                first_lst = list(range(el + 1, 10))
                second_list = list(np.full(len(first_lst), el))
                res[0] = res[0] + second_list
                res[1] = res[1] + first_lst
            edge_index = torch.tensor(res, dtype=torch.long)
            data = Data(img=img, pos=pos, name=str(idx), edge_index=edge_index, keypoints=keypoint_list,rgb_img=image_rgb)
            data_list.append(data)
        imgs = [data.rgb_img.T for data in data_list]
        loader = DataLoader(imgs, self.batch_size, shuffle=False)
        data_list = self.load_vgg_processed_inputs(loader, data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])
        return


def main():
    wl_ds = FacialDataset()
    print(wl_ds[0])

    return


if __name__ == '__main__':
    main()






