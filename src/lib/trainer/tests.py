import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.supervised import Supervised
from datasets.rdm import Rdms
from networks import resnet, densenet, normal
from utils.data import WrappedDataLoader
from utils import io, logger, metrics


def create_tester(exp_id, mode, model_type, bs=32, size=30):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    # Create a trainer depending on the mode
    if mode == 'supervised':
        test_ds = Supervised('test', transforms=tfms, size=size)
        loss_func = torch.nn.CrossEntropyLoss()
        if model_type == 'resnet':
            model = resnet.Telnet()
        elif model_type == 'desne':
            model = densenet.Telnet()
        elif model_type == 'normal':
            model = normal.Telnet()
        metric = metrics.cross_acc
    elif mode == 'siamese':
        test_ds = Rdms('test', transforms=tfms, size=size)
        loss_func = metrics.RMSELoss()
        if model_type == 'resnet':
            model = resnet.Siameserdm()
        elif model_type == 'desne':
            model = densenet.Siameserdm()
        elif model_type == 'normal':
            model = normal.Siameserdm()
        metric = metrics.rmse_acc
    log = logger.Logger(exp_id, mode, model_type, size)
    test_dl = WrappedDataLoader(DataLoader(
        test_ds, batch_size=bs * 2, num_workers=4))
    model = io.load(model, log.full_path, size)
    return Tester(model, test_dl, loss_func, log, metric)


class Tester():
    def __init__(self, model, test_dl, logger, metric):
        self.model = model
        self.test_dl = test_dl
        self.logger = logger
        self.metric = metric
        self.model.to(self.device)

    def fit(self, epochs):
        self.model.eval()
        with torch.no_grad():
            tot_acc = 0.
            for xb, yb in self.test_dl:
                pred = self.model(xb)
                tot_acc += self.metric(pred, yb)
        nv = len(self.test_dl)
        acc = tot_acc / nv

        print('Acc: {}'.format(acc))
        self.logger.add_results(acc)
