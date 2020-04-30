import torch


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

    def forward(self, yhat, y): return torch.sqrt(self.mse(yhat, y) + self.eps)
