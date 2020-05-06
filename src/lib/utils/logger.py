from pathlib import Path
import torch
from torchsummary import summary
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os


class Logger():
    def __init__(self, exp_id, mode, model_type, size, lr, bs, version):
        self.exp_id = exp_id
        self.mode = mode
        self. model_type = model_type
        self.size = size
        self.lr = lr
        self.bs = bs
        self.ver = version
        self.base = Path(Path.cwd(), '../Logs')
        if self.ver is not None:
            self.full_path = self.base / self.exp_id / \
                self.mode / str(self.ver) / self.model_type
        else:
            self.full_path = self.base / self.exp_id / self.mode / \
                self.model_type
        print(self.full_path)
        self.loss_list = []
        os.makedirs(self.full_path, exist_ok=True)

    def create_data(self):
        f = open(self.full_path / 'data.txt', 'w+')
        f.write('Mode: {}\n'.format(self.mode))
        f.write('Model: {}\n'.format(self.model_type))
        f.write('Exp ID: {}\n'.format(self.exp_id))
        f.write('Epochs: {}\n'.format(self.epochs))
        f.write('lr: {}\n'.format(self.lr))
        f.write('Data size: {}\n'.format(self.size))
        f.write('Batch size: {}\n'.format(self.bs))
        f.write('RDM version: {}\n'.format(self.ver))
        f.close()

    def log_summary(self, model, input_shape):
        sum_file = open(self.full_path / 'summary.txt', 'w+')
        sum_file.write(
            str(summary(model, input_data=input_shape, verbose=0, depth=5)))
        sum_file.close()

    def log(self, loss_list):
        self.loss_list.append(loss_list)

    def log_info(self, epochs, lr):
        self.epochs = epochs
        self.create_data()

        print('\n\nModel: {}; Mode:{};  Size: {}\nEpochs: {}; lr: {}\n\n'
              .format(
                  self.model_type, self.mode, self.size, epochs, lr))

    def plot(self):
        a = np.arange(1, len(self.loss_list) + 1)
        loss = np.array(self.loss_list)
        plt.plot(a, loss[:, 0], label='Train loss')
        plt.plot(a, loss[:, 1], label='Val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('{}; {}; {}'.format(self.exp_id, self.mode, self.model_type))
        plt.legend()
        plt.savefig(self.full_path / 'loss_{}.png'.format(self.size))
        plt.close()
        plt.plot(a, loss[:, 2] * 100, label='Accuracy')
        plt.title('{}; {}; {}'.format(self.exp_id, self.mode, self.model_type))
        plt.savefig(self.full_path / 'acc_{}.png'.format(self.size))
        plt.close()

    def add_result(self, val):
        f = open(self.full_path / 'results.txt', 'a+')
        f.write('Size: {}, Accuarcy: {}, timestamp: {}'.format(
            self.size, val, datetime.now()))
        f.close()

    def done(self):
        torch.save(self.loss_list, self.full_path / 'loss.pth')
        self.plot()
