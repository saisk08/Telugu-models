from torch.utils.data import Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
from scipy.spatial.distance import squareform


class Rdms(Dataset):
    '''Dataset class for telugu chars; siamese learning with rdms'''

    def __init__(self, mode=None, custom=True, transforms=None, size=None):
        super().__init__()
        self.path = Path('../../../data/siamese')

        if size is None:
            f = open(self.path / 'chars.pkl', 'rb')
        else:
            f = open(self.path / 'char_{}.pkl'.format(size), 'rb')
        self.train, self.val, self.test = pickle.load(f)
        f.close()

        if custom:
            f = open(self.path / 'rdms_custom.pkl', 'rb')
        else:
            f = open(self.path / 'rdm_minmax.pkl', 'rb')
        self.rdm = pickle.load(f)

        # Because one char is not available; index = 9
        self.rdm = squareform(self.rdm)
        self.rdm = np.delete(self.rdm, 9, 0)
        self.rdm = np.delete(self.rdm, 9, 1)
        self.rdm = squareform(self.rdm)

        self.mode = mode
        self.tfms = transforms

    def __len__(self):
        if self.mode == 'train':
            return np.multiply(*self.train[[0, 2]])
        if self.mode == 'val':
            return np.multiply(*self.val[[0, 2]])
        if self.mode == 'test':
            return np.multiply(*self.test[[0, 2]])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.numpy()
        if self.mode == 'train':
            cat = index // self.train.shape[2]
            ids = index // self.train.shape[0]
            img1, img2, label = self.train[cat, 0, ids], \
                self.train[cat, 1, ids], self.rdm[cat]
        if self.mode == 'val':
            cat = index // self.val.shape[2]
            ids = index // self.val.shape[0]
            img1, img2, label = self.val[cat, 0, ids], \
                self.val[cat, 1, ids], self.rdm[cat]
        if self.mode == 'test':
            cat = index // self.test.shape[2]
            ids = index // self.test.shape[0]
            img1, img2, label = self.test[cat, 0, ids], \
                self.test[cat, 1, ids], self.rdm[cat]

        img1, img2 = img1.reshape(-1, 32, 32), img2.reshape(-1, 32, 32)
        if self.tfms is not None:
            img1 = self.tfms(img1)
            img2 = self.tfms(img2)

        return img1, img2, label
