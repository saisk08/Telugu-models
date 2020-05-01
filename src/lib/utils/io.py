import torch
from pathlib import Path
import os


def save(model, p, size): return torch.save(
    model.state_dict(), p / 'model_{}.pth'.format(size))


def load(model, sup_id, model_type, size):
    p = Path(Path.cwd, '../../../Logs', sup_id, model_type)
    return model.load_state_dict(
        torch.load(p / 'model_{}.pth'.format(size)))
