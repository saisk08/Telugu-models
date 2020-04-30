from pathlib import Path
import os
import pickle
from torchsummary import summary
from matplotlib import pyplot as plt
from datetime import datetime


class Logger():
    def __init__(self, exp_id, mode, model_type, test=False):
        self.exp_id = exp_id
        self.mode = mode
        self. model_type = model_type
        self.base = Path(
            '../../../Logs') if not test else Path('../../../Results')
        self.full_path = self.base / self.exp_id / self.mode / self.model_type
        self.loss_list = []
        os.makedirs(self.full_path)

    def log_summary(self, model, input_shape):
        sum_file = open(self.full_path / 'summary.txt', 'w+')
        sum_file.write(str(summary(model, input_data=input_shape, verbose=0)))
        sum_file.close()

    def log(self, loss_list):
        self.loss_list.append(loss_list)

    def plot(self):
        plt.plot(self.loss_list[:, 0],
                 self.loss_list[:, 1], label='Train loss')
        plt.plot(self.loss_list[:, 0],
                 self.loss_list[:, 2], label='Val loss')
        plt.plot(self.loss_list[:, 0],
                 self.loss_list[:, 3], label='Accuracy')
        plt.savefig(self.full_path / 'plot.png')
        plt.title('{}; {}; {}'.format(self.exp_id, self.mode, self.model_type))
        plt.legend()
        plt.close()

    def add_result(self, val):
        f = open(self.full_path / 'results.txt', 'a+')
        f.write('Accuarcy: {}, timestamp: {}'.format(val, datetime.now()))
        f.close()

    def done(self):
        pkl = open(self.full_path / 'loss.pkl', 'wb')
        self.plot()
        pickle.dump(self.loss_list, pkl)
        pkl.close()
