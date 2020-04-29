import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import DataLoader


from pycocotools.coco import COCO
import skimage.io as io
import torchvision
from torch_geometric.data import InMemoryDataset, Data


# dataset interface takes the ids of the COCO classes
class COCODataset(InMemoryDataset):
    categories = ['duck']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    def __init__(self, root='./data/coco_processed/', category='face', transform=None, pre_transform=None):
        self.root = root
        self.category = category
        interf = os.path.join(self.root, 'coco')
        file = os.path.join(interf, "person_keypoints_train2017.json")
        self.coco_interface = COCO(file)
        super(COCODataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'coco_processed/coco/')

    @property
    def processed_file_names(self):
        return ['./processed']

    @property
    def raw_file_names(self):
        return [category.capitalize() for category in self.categories]


    def load_img(self, idx):
        im = np.array(io.imread(self.img_data[idx]['coco_url']))
        im = self.transform(im)
        return im

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
        person_cats = self.coco_interface.getCatIds()
        im_ids = self.coco_interface.getImgIds(catIds=person_cats)
        self.img_data = self.coco_interface.loadImgs(im_ids)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data_list = []
        print(len(self.img_data))
        for count,img in enumerate(self.img_data):
            if count %100 ==0:
              print(count)
            im_id = img['id']
            annIds = self.coco_interface.getAnnIds(imgIds=im_id, catIds=[1], iscrowd=None)
            # get the annotations
            anns = self.coco_interface.loadAnns(annIds)

            poss, ys = [], []
            # for a in anns:
            kp = anns[0]['keypoints']
            kp = np.reshape(np.array(kp), (17, 3))
            x = kp[:, 0]
            y = kp[:, 1]
            poss += [x, y]
            y = torch.tensor(ys, dtype=torch.long)
            pos = torch.tensor(poss, dtype=torch.float).view(-1, 2)
            img_arr = Image.fromarray(np.array(io.imread(img['coco_url'])))
            pos[:, 0] = pos[:, 0] * 256.0 / (img_arr.size[0])
            pos[:, 1] = pos[:, 1] * 256.0 / (img_arr.size[1])
            try:
                img_arr = img_arr.resize((256, 256), resample=Image.BICUBIC)
                img_arr = transform(img_arr)
                img_name = img['coco_url']

                data = Data(img=img_arr, pos=pos, name=img_name)
                data_list.append(data)
            except:
                print(img_arr)
        imgs = [data.img for data in data_list]
        loader = DataLoader(imgs, self.batch_size, shuffle=False)
        data_list = self.load_vgg_processed_inputs(loader, data_list)
        filter_out = [data for data in data_list if data.x.shape[0] == 10]
        torch.save(self.collate(filter_out), self.processed_paths[0])


def main():
    wl_ds = COCODataset()
    print(wl_ds[0])
    return


if __name__ == '__main__':
    main()