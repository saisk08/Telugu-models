import torch


def save(model, p): return torch.save(model.state_dict(), p / 'model.pth')


def load(model, p): return model.load_state_dict(torch.load(p / 'model.pth'))
