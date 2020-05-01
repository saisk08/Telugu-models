import torch


def supervised(x, y):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return x.view(-1, 1, 32, 32).to(device), y.to(device)


def siamese(a, b, y):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return a.view(-1, 1, 32, 32).to(device), b.view(-1, 1, 32, 32).to(device), y.to(device)


class WrappedDataLoader:
    def __init__(self, dl, mode):
        self.dl = dl
        if mode == 'supervised':
            self.func = supervised
        else:
            self.func = siamese

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
