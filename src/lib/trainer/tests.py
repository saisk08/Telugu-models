import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.supervised import Supervised
from networks import resnet, densenet, normal
from utils.data import WrappedDataLoader
from utils import io, logger, metrics


def create_examiner(exp_id, mode, model_type, bs=32, size=30):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    # Create a examiner depending on the mode
    test_ds = Supervised('test', transforms=tfms, size=size)
    metric = metrics.cross_acc
    if model_type == 'resnet':
        model = resnet.Telnet()
    elif model_type == 'desne':
        model = densenet.Telnet()
    elif model_type == 'normal':
        model = normal.Telnet()
    loss_func = torch.nn.CrossEntropyLoss()
    log = logger.Logger(exp_id, 'tested', model_type, size)
    test_dl = WrappedDataLoader(DataLoader(
        test_ds, batch_size=bs * 2, num_workers=4))
    model = io.load(model, exp_id, model_type, size, mode)
    return Examiner(model, test_dl, loss_func, log, metric)


class Examiner():
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
