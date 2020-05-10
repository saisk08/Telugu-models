import torch
import torch.nn.functional as F


def cross_acc(out, yb): return (
    torch.argmax(out, dim=1) == yb).float().mean()


def rmse_acc(out, yb):
    loss = RMSELoss()
    return 1 - loss(out, yb)


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, y1, y2, y):
        yhat = F.pairwise_distance(y1, y2)
        return torch.sqrt(self.mse(yhat, y) + self.eps)


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y1, y2, y):
        return torch.nn.MSELoss(F.pairwise_distance(y1, y2), y)


class RDLoss(torch.nn.Module):
    '''Modified version of Contrastive loss, Same as L1 loss'''

    def __init__(self):
        super().__init__()

    def forward(self, out1, out2, rdm):
        return torch.mean(rdm - F.pairwise_distance(out1, out2))
