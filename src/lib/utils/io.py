import torch


def save(model, p, size): return torch.save(
    model.state_dict(), p / 'model_{}.pth'.format(size))


def load(model, p, size): return model.load_state_dict(
    torch.load(p / 'model_{}.pth'.format(size)))
