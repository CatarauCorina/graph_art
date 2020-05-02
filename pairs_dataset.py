from torch_geometric.data import DataLoader, Dataset
from willow_ip import WillowDataset
import numpy as np
import torch
import itertools
import random
from pascal_voc_ip import PascalVOCDataset



class PairsDS(Dataset):

    def __init__(self, ds_type='willow', mode='train'):
        if ds_type == 'willow':
            ds = WillowDataset()
            val_size = 10

        elif ds_type == 'facial':
            ds = FacialDataset()
            val_size = 50
        self.nr_keypoints = len(ds[0].keypoints[0])
        self.train_dataset = ds[:(len(ds) - val_size)]
        self.val_dataset = ds[(len(ds) - val_size):len(ds)]
        self.pairs_train = list(itertools.product(self.train_dataset, repeat=2))
        self.pairs_valid = list(itertools.product(self.val_dataset, repeat=2))
        # self.pairs_train = list(zip(self.train_dataset[0::2], self.train_dataset[1::2]))
        # self.pairs_valid = list(zip(self.val_dataset[0::2], self.val_dataset[1::2]))
        if mode == 'train':
            self.pairs = self.pairs_train
        else:
            self.pairs = self.pairs_valid
        return

    def create_permutation_matrix(self, val_x, val_y):
        perm_mat = np.zeros([self.nr_keypoints, self.nr_keypoints], dtype=np.float32)
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
        return perm_mat

    def __getitem__(self, index):
        pair = self.pairs[index]
        perm_matrix = self.create_permutation_matrix(pair[0], pair[1])
        n1_gt, n2_gt = len(pair[0].keypoints[0]), len(pair[1].keypoints[0])
        perm_matrix = np.array(perm_matrix)

        pair_data = {
            'pair': pair,
            'perm_matrix': perm_matrix.tolist(),
            'n1_gt': n1_gt,
            'n2_gt': n2_gt
        }

        return pair_data

    def __len__(self):
        return len(self.pairs)