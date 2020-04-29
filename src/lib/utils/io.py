import torch


def save(model, p): return torch.save(model.state_dict(), p)


def load(model, p): return model.load_state_dict(torch.load(p))
