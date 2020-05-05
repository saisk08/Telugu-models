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


def create_trainer(exp_id, mode, model_type, lr=3e-3, bs=32, size=30, version=None):
    tfms = transforms.Compose([
        transforms.ToTensor()
    ])
    # Create a trainer depending on the mode
    if mode == 'supervised':
        train_ds = Supervised('train', transforms=tfms, size=size)
        valid_ds = Supervised('val', transforms=tfms, size=size)
        train_dl, valid_dl = get_dls(train_ds, valid_ds)
        train_dl = WrappedDataLoader(train_dl, mode)
        valid_dl = WrappedDataLoader(valid_dl, mode)
        loss_func = nn.CrossEntropyLoss()
        metric = metrics.cross_acc
    elif mode == 'siamese':
        if version == 1:
            train_ds = Rdms('train', transforms=tfms, size=size, version1=True)
            valid_ds = Rdms('val', transforms=tfms, size=size)
        elif version == 2:
            train_ds = Rdms('train', transforms=tfms,
                            size=size, version1=False)
            valid_ds = Rdms('val', transforms=tfms, size=size, version1=False)
        train_dl, valid_dl = get_dls(train_ds, valid_ds)
        train_dl = WrappedDataLoader(train_dl, mode)
        valid_dl = WrappedDataLoader(valid_dl, mode)
        loss_func = metrics.RMSELoss()
        metric = None

    if model_type == 'resnet':
        model = resnet.Telnet()
    elif model_type == 'desne':
        model = densenet.Telnet()
    elif model_type == 'normal':
        model = normal.Telnet()
    log = logger.Logger(exp_id, mode, model_type, size, lr, bs, version)
    return Trainer(model, train_dl, valid_dl, loss_func, lr, log, metric, mode)


class Trainer():
    def __init__(self, model, train_dl, valid_dl, loss_func, lr, logger,
                 metric, mode):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_func = loss_func
        self.logger = logger
        self.metric = metric
        self.lr = lr
        self.logger.log_summary(self.model, (1, 32, 32))
        self.opt = optim.SGD(self.model.parameters(), lr=lr,
                             momentum=0.9, weight_decay=0.02)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit_supervised(self, epochs):
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
            print('Epoch: {}, train loss: {:.6f}, val loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, loss, val_loss, acc * 100))
            self.logger.log([loss, val_loss, acc])

    def fit_siamese(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for x1b, x2b, rdm in self.train_dl:
                out1 = self.model(x1b)
                out2 = self.modle(x2b)
                loss = self.loss_func(out1, out2, rdm)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            self.model.eval()
            with torch.no_grad():
                tot_loss, tot_acc = 0., 0.
                for x1b, x2b, rdm in self.val_dl:
                    out1 = self.model(x1b)
                    out2 = self.modle(x2b)
                    temp = self.loss_func(out1, out2, rdm)
                    tot_loss += temp
                    tot_acc += 1 - temp
            nv = len(self.valid_dl)
            val_loss = tot_loss / nv
            acc = tot_acc / nv
            print('Epoch: {}, train loss: {:.6f}, val loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, loss, val_loss, acc * 100))
            self.logger.log([loss, val_loss, acc])

    def fit(self, epochs):
        self.logger.log_info(epochs, self.lr)
        if mode == 'supervised':
            self.fit_supervised()
        elif mode == 'siamese':
            self.fit_siamese()

        self.logger.done()
        io.save(self.model, self.logger.full_path, self.logger.size)


def create_finetuner(exp_id, sia_id, model_type, version, lr=3e-3, bs=32, size=30):
    mode = 'supervised'
    if model_type == 'resnet':
        model = resnet.Telnet()
    elif model_type == 'desne':
        model = densenet.Telnet()
    elif model_type == 'normal':
        model = normal.Telnet()

    model = io.load(model, sia_id, model_type, size,
                    mode='siamese', version=version)
    train_ds = Supervised('train', transforms=tfms, size=size)
    valid_ds = Supervised('val', transforms=tfms, size=size)
    train_dl, valid_dl = get_dls(train_ds, valid_ds)
    train_dl = WrappedDataLoader(train_dl, mode)
    valid_dl = WrappedDataLoader(valid_dl, mode)
    loss_func = nn.CrossEntropyLoss()
    metric = metrics.cross_acc
    log = logger.Logger(exp_id, 'tuned', model_type, size, lr, bs, None)
    return Finetuner(model, train_dl, valid_dl, loss_func, lr, log, metric)


class Finetuner():
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
        self.opt = optim.SGD(self.model.parameters(), lr=lr,
                             momentum=0.9, weight_decay=0.02)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, epochs):
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
            print('Epoch: {}, train loss: {:.6f}, val loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, loss, val_loss, acc * 100))
            self.logger.log([loss, val_loss, acc])

        self.logger.done()
        io.save(self.model, self.logger.full_path, self.logger.size)
