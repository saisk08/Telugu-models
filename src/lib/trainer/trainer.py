from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
import torch
from datasets.supervised import Supervised
from datasets.rdm import Rdms
from utils.data import WrappedDataLoader
from networks import resnet, densenet, normal
from utils import io, logger, metrics


def get_dls(train_ds, valid_ds, bs=16):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
        DataLoader(valid_ds, batch_size=bs * 2, num_workers=0),
    )


def create_trainer(exp_id, mode, model_type, lr=3e-3, bs=32, size=30):
    tfms = transforms.Compose([
        transforms.ToTensor()
    ])
    # Create a trainer depending on the mode
    if mode == 'supervised':
        train_ds = Supervised('train', transforms=tfms, size=size)
        valid_ds = Supervised('val', transforms=tfms, size=size)
        train_dl, valid_dl = get_dls(train_ds, valid_ds)
        train_dl = WrappedDataLoader(train_dl)
        valid_dl = WrappedDataLoader(valid_dl)
        loss_func = nn.CrossEntropyLoss()
        if model_type == 'resnet':
            model = resnet.Telnet()
        elif model_type == 'desne':
            model = densenet.Telnet()
        elif model_type == 'normal':
            model = normal.Telnet()
        metric = metrics.cross_acc
    elif mode == 'siamese':
        train_ds = Rdms('train', transforms=tfms, size=size)
        valid_ds = Rdms('val', transforms=tfms, size=size)
        train_dl, valid_dl = get_dls(train_ds, valid_ds)
        train_dl = WrappedDataLoader(train_dl)
        valid_dl = WrappedDataLoader(valid_dl)
        loss_func = metrics.RMSELoss()
        if model_type == 'resnet':
            model = resnet.Siameserdm()
        elif model_type == 'desne':
            model = densenet.Siameserdm()
        elif model_type == 'normal':
            model = normal.Siameserdm()
        metric = None
    log = logger.Logger(exp_id, mode, model_type, size)
    return Trainer(model, train_dl, valid_dl, loss_func, lr, log, metric)


class Trainer():
    def __init__(self, model, train_dl, valid_dl, loss_func, lr, logger,
                 metric):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_func = loss_func
        self.logger = logger
        self.metric = metric
        self.lr = lr
        self.logger.log_summary(self.model, (1, 32, 32))
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, epochs):
        self.logger.log_info(epochs, self.lr)
        for epoch in range(epochs):
            self.model.train()
            for xb, yb in self.train_dl:
                loss = self.loss_func(self.model(xb), yb)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            self.model.eval()
            with torch.no_grad():
                tot_loss, tot_acc = 0., 0.
                for xb, yb in self.valid_dl:
                    pred = self.model(xb)
                    temp = self.loss_func(pred, yb)
                    tot_loss += temp
                    tot_acc += self.metric(pred,
                                           yb) if self.metric else 1 - temp
            nv = len(self.valid_dl)
            val_loss = tot_loss / nv
            acc = tot_acc / nv
            print('Epoch: {}, train loss: {:.4f}, val loss: {:.4f}, Acc: {:.4f}'.format(
                epoch + 1, loss, val_loss, acc * 100))
            self.logger.log([loss, val_loss, acc])

        self.logger.done()
        io.save(self.model, self.logger.full_path, self.logger.size)
