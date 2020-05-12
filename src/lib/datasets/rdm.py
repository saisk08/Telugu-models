from torch.utils.data import Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
import os


class Rdms(Dataset):
    '''Dataset class for telugu chars; siamese learning with rdms'''

    def __init__(self, mode=None, custom=True, version1=True, transforms=None,
                 size=None):
        super().__init__()
        this_file = os.path.dirname(__file__)
        self.path = Path('../../../data/siamese')
        self.path = Path(os.path.join(this_file, self.path))
        if size is None:
            f = open(self.path / 'chars.pkl', 'rb')
        else:
            f = open(self.path / 'chars_{}.pkl'.format(size), 'rb')
        self.train, self.val, self.test = pickle.load(f)
        f.close()
        self.train, self.val, self.test = self.train.astype(
            np.float32), self.val.astype(np.float32), \
            self.test.astype(np.float32)
        # Because ta is not avaible
        if custom:
            f = open(self.path / 'rdms_custom_w.pkl', 'rb')
        else:
            f = open(self.path / 'rdm_minmax_w.pkl', 'rb')
        self.rdm = pickle.load(f)
        f.close()
        self.rdm = self.rdm[0] if version1 else self.rdm[1]
        self.version = 1 if version1 else 2

        # Because one char is not available; index = 9

        self.mode = mode
        self.tfms = transforms

    def __len__(self):
        if self.mode == 'train':
            return self.train.shape[0] * self.train.shape[2]
        if self.mode == 'val':
            return self.val.shape[0] * self.val.shape[2]
        if self.mode == 'test':
            return self.test.shape[0] * self.test.shape[2]

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
            label = torch.tensor(label).float()

        return img1, img2, label
