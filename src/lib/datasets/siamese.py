from torch.utils.data import Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
import os


class Siamese(Dataset):
    '''Dataset class for telugu chars; siamese learning without rdms'''

    def __init__(self, mode=None, transforms=None, size=None):
        super().__init__()
        this_file = os.path.dirname(__file__)
        self.path = Path('../../../data/siamese')
        self.path = Path(os.path.join(this_file, self.path))
        if size is None:
            f = open(self.path / 'chars.pkl', 'rb')
        else:
            f = open(self.path / 'chars_{}.pkl'.format(size), 'rb')
        self.train, self.val, self.test = pickle.load(f)
        self.train, self.val, self.test = self.train.astype(
            np.float32), self.val.astype(np.float32), \
            self.test.astype(np.float32)
        f.close()
        self.mode = mode
        self.tfms = transforms

    def __len__(self):
        if self.mode == 'train':
            return np.multiply(*self.train.shape[[0, 2]])
        if self.mode == 'val':
            return np.multiply(*self.val.shape[[0, 2]])
        if self.mode == 'test':
            return np.multiply(*self.test.shape[[0, 2]])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.numpy()
        if self.mode == 'train':
            cat = index // self.train.shape[2]
            ids = index // self.train.shape[0]
            img1, img2, label = self.train[cat, 0, ids], \
                self.train[cat, 1, ids], cat
        if self.mode == 'val':
            cat = index // self.val.shape[2]
            ids = index // self.val.shape[0]
            img1, img2, label = self.val[cat, 0, ids], \
                self.val[cat, 1, ids], cat
        if self.mode == 'test':
            cat = index // self.test.shape[2]
            ids = index // self.test.shape[0]
            img1, img2, label = self.test[cat, 0, ids], \
                self.test[cat, 1, ids], cat

        img1, img2 = img1.reshape(-1, 32, 32), img2.reshape(-1, 32, 32)
        if self.tfms is not None:
            img1 = self.tfms(img1)
            img2 = self.tfms(img2)

        return img1, img2, label
