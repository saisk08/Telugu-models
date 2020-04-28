from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
import torch
import numpy as np


def preprocess(x, y):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return x.view(-1, 1, 32, 32).to(device), y.to(device)


def create_trainer(mode, lr=3e-3, bs=32):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])
    # Create train sets
    # Create dataloaders; wrap it
    # Get loss and model
    train_dl = DataLoader(trainset, batch_size=bs,
                          num_workers=4, transforms=tfms)
    valid_dl = DataLoader(validset, batch_size=bs, num_workers=4)
    if mode == 'supervised':
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.MSELoss()
    return Trainer(model, train_dl, valid_dl, loss_func, lr)


class Trainer():
    def __init__(self, model, train_dl, valid_dl, loss_func, lr):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_func = loss_func
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.device =
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
                self.loss_batch(xb.to(self.device), yb.to(self.device))

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(*[self.loss_batch(xb.to(self.device),
                                                     yb.to(self.device))
                                     for xb, yb in self.valid_dl])

            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print('Epoch: {}, val loss: {}'.format(epoch + 1, val_loss))
