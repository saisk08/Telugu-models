from torch.utils.data import Dataset
from pathlib import Path
import pickle
import torch
import numpy as np


class Supervised(Dataset):
    '''Dataset class for telugu chars; supervised version'''

    def __init__(self, mode=None, transforms=None, size=None):
        super().__init__()
        self.path = Path('../../../data/supervised')
        if size is None:
            f = open(self.path / 'chars.pkl', 'rb')
        else:
            f = open(self.path / 'char_{}.pkl'.format(size), 'rb')
        self.train, self.val, self.test = pickle.load(f)
        f.close()
        self.mode = mode
        self.tfms = transforms

    def __len__(self):
        if self.mode == 'train':
            return np.multiply(*self.train[:2])
        if self.mode == 'val':
            return np.multiply(*self.val[:2])
        if self.mode == 'test':
            return np.multiply(*self.test[:2])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.numpy()
        if self.mode == 'train':
            cat = index // self.train.shape[1]
            ids = index // self.train.shape[0]
            img, label = self.train[cat, ids], cat
        if self.mode == 'val':
            cat = index // self.val.shape[1]
            ids = index // self.val.shape[0]
            img, label = self.val[cat, ids], cat
        if self.mode == 'test':
            cat = index // self.test.shape[1]
            ids = index // self.test.shape[0]
            img, label = self.test[cat, ids], cat

        img = img.reshape(-1, 32, 32)
        if self.tfms is not None:
            img = self.tfms(img)

        return img, label
