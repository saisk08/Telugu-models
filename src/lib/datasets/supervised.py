from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import torch


class SupervisedTelugu(Dataset):
    '''Dataset class for telugu chars; supervised version'''

    def __init__(self, train=False, val=False, test=False, transforms=None):
        super().__init__()
        self.path = Path('../../../data')
        self.f = open(self.path / 'UHTelPCC.pkl', 'rb')
        self.dump = pickle.load(self.f, encoding='latin-1')
        (self.train_x, self.train_y), (self.val_x,
                                       self.val_y), (self.test_x, self.test_y) = self.dump
        self.train = train
        self.val = val
        self.test = test
        self.tfms = transforms

    def __len__(self):
        if self.train:
            return self.train_y.shape[0]
        if self.val:
            return self.val_y.shape[0]
        if self.test:
            return self.test_y.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.train:
            img, label = self.train_x[index], self.train_y[index]
        if self.val:
            img, label = self.train_x[index], self.train_y[index]
        if self.test:
            img, label = self.train_x[index], self.train_y[index]

        if self.tfms is not None:
            img = self.tfms(img)

        return img, label
