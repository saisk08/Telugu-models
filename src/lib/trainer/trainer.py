from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
import torch
import numpy as np
from datasets.supervised import Supervised
from datasets.rdm import Rdms
from utils.data import WrappedDataLoader
from networks import resnet, densenet, normal


def preprocess(x, y):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return x.view(-1, 1, 32, 32).to(device), y.to(device)


def get_dls(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4),
        DataLoader(valid_ds, batch_size=bs * 2, num_workers=4),
    )


def create_trainer(mode, model_type, lr=3e-3, bs=32, size=30):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    # Create a trainer depending on the mode
    if mode == 'supervised':
        train_ds = Supervised('train', transforms=tfms, size=size)
        valid_ds = Supervised('val', transforms=tfms, size=size)
        train_dl, valid_dl = get_dls(train_ds, valid_ds)
        train_dl = WrappedDataLoader(train_dl, preprocess)
        valid_dl = WrappedDataLoader(valid_dl, preprocess)
        loss_func = nn.CrossEntropyLoss()
        if model_type == 'resnet':
            model = resnet.Telnet()
        elif model_type == 'desne':
            model = densenet.Telnet()
        elif model_type == 'normal':
            model = normal.Telnet()
    elif mode == 'siamese':
        train_ds = Rdms('train', transforms=tfms, size=size)
        valid_ds = Rdms('val', transforms=tfms, size=size)
        train_dl, valid_dl = get_dls(train_ds, valid_ds)
        train_dl = WrappedDataLoader(train_dl, preprocess)
        valid_dl = WrappedDataLoader(valid_dl, preprocess)
        loss_func = nn.MSELoss()
        if model_type == 'resnet':
            model = resnet.Siameserdm()
        elif model_type == 'desne':
            model = densenet.Siameserdm()
        elif model_type == 'normal':
            model = normal.Siameserdm()

    return Trainer(model, train_dl, valid_dl, loss_func, lr)


class Trainer():
    def __init__(self, model, train_dl, valid_dl, loss_func, lr):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_func = loss_func
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)

    def loss_batch(self, xb, yb):
        loss = self.loss_func(self.model(xb), yb)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item(), len(xb)

    def fit(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for xb, yb in self.train_dl:
                self.loss_batch(xb, yb)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(*[self.loss_batch(xb, yb)
                                     for xb, yb in self.valid_dl])

            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print('Epoch: {}, val loss: {}'.format(epoch + 1, val_loss))
