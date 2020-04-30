from pathlib import Path
import os
import torch
from torchsummary import summary
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np


class Logger():
    def __init__(self, exp_id, mode, model_type, size):
        self.exp_id = exp_id
        self.mode = mode
        self. model_type = model_type
        self.size = size
        base = os.path.dirname(__file__)
        self.base = Path(os.path.join(base, '../../../Logs'))
        self.full_path = self.base / self.exp_id / self.mode / self.model_type
        self.loss_list = []
        os.makedirs(self.full_path, exist_ok=True)

    def log_summary(self, model, input_shape):
        sum_file = open(self.full_path / 'summary.txt', 'w+')
        sum_file.write(
            str(summary(model, input_data=input_shape, verbose=0, depth=4)))
        sum_file.close()

    def log(self, loss_list):
        self.loss_list.append(loss_list)

    def log_info(self, epochs, lr):
        print('\n\nModel: {}; Mode:{};  Size: {}\nEpochs: {}; lr: {}\n\n'.format(
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
        plt.plot(a, loss[:, 2], label='Accuracy')
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
