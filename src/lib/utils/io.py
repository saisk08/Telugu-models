import torch
from pathlib import Path
import os


def save(model, p, size): return torch.save(
    model.state_dict(), p / 'model_{}.pth'.format(size))


def load(model, exp_id, model_type, size, mode='supervised', version=None):
    if version is not None:
        p = Path(Path.cwd, '../../../Logs', exp_id, mode, version, model_type)
    else:
        p = Path(Path.cwd, '../../../Logs', exp_id, mode, model_type)
    return model.load_state_dict(
        torch.load(p / 'model_{}.pth'.format(size)))
