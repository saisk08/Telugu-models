from pathlib import Path
import torch
from torchsummary import summary
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os
from tabulate import tabulate


class Logger():
    def __init__(self, exp_id, mode, model_type, size, lr, bs, version,
                 sia_id=None):
        self.exp_id = exp_id
        self.sia_id = sia_id if sia_id is not None else exp_id
        self.mode = mode
        self. model_type = model_type
        self.size = size
        self.lr = lr
        self.bs = bs
        self.ver = version
        self.base = Path(Path.cwd(), '../Logs')
        if self.ver is not None:
            self.full_path = self.base / self.exp_id / \
                self.mode / str(self.ver) / self.model_type / str(self.size)
            self.load_path = self.base / self.sia_id / \
                'siamese' / str(self.ver) / self.model_type / str(self.size)
        else:
            self.full_path = self.base / self.exp_id / self.mode / \
                self.model_type / str(self.size)
            self.load_path = self.base / self.sia_id / 'siamese' / \
                self.model_type / str(self.size)
        self.loss_list = []
        os.makedirs(self.full_path, exist_ok=True)

    def log_summary(self, model, input_shape):
        sum_file = open(self.full_path / 'summary.txt', 'w+')
        sum_file.write(
            str(summary(model, input_data=input_shape, verbose=0, depth=5)))
        sum_file.close()

    def log(self, loss_list):
        self.loss_list.append(loss_list)

    def log_info(self, epochs, lr):
        self.epochs = epochs
        f = open(self.full_path / 'data.txt', 'w+')
        data = []
        data.append(['Exp ID', self.exp_id])
        data.append(['Model type', self.model_type])
        data.append(['Mode', self.mode])
        if self.ver is not None:
            data.append(['Version', self.ver])
        data.append(['Size', self.size])
        data.append(['Epochs', epochs])
        data.append(['Learing rate', lr])
        data.append(['Batch size', self.bs])
        a = tabulate(data)
        f.write(a)
        f.close()
        print('\n\n' + a + '\n\n')
        return

    def plot(self):
        a = np.arange(1, len(self.loss_list) + 1)
        loss = np.array(self.loss_list)
        plt.plot(a, loss[:, 0], label='Train loss')
        plt.plot(a, loss[:, 1], label='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('{}; {}; {}'.format(self.exp_id, self.mode, self.model_type))
        plt.legend()
        plt.savefig(self.full_path / 'loss_graph.png')
        plt.close()
        if self.mode != 'siamese':
            plt.plot(a, loss[:, 2], label='Accuracy')
            plt.title('{}; {}; {}'.format(
                self.exp_id, self.mode, self.model_type))
            plt.savefig(self.full_path / 'acc_graph.png')
            plt.close()

    def add_result(self, val):
        f = open(self.full_path / 'results.txt', 'a+')
        f.write('Size: {}, Accuarcy: {}, timestamp: {}'.format(
            self.size, val, datetime.now()))
        f.close()

    def done(self):
        torch.save(self.loss_list, self.full_path / 'loss_vals.pth')
        self.plot()
