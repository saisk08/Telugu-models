import torch


def supervised(x, y):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return x.view(-1, 1, 32, 32).to(device), y.to(device)


def siamese(a, b, y):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    return a.view(-1, 1, 32, 32).to(device), \
        b.view(-1, 1, 32, 32).to(device), \
        y.to(device)


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


class SiameseSampler():
    def __init__(self, ds, bs, shuffle=False):
        self.n, self.bs, self.shuffle = len(ds), bs, shuffle

    def __iter__(self):
        self.idxs1 = torch.randperm(self.n)
        self.idxs2 = torch.randperm(self.n)
        for i in range(0, self.n, self.bs):
            yield self.idxs1[i:i+self.bs], self.idxs2[i:i+self.bs]
