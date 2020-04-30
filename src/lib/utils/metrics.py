import torch


def cross_acc(out, yb): return (
    torch.argmax(out, dim=1) == yb).float().mean()
