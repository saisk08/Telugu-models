from pathlib import Path
import os
import pickle
from torchsummary import summary


class Logger():
    def __init__(self, exp_id, mode, model_type):
        self.exp_id = exp_id
        self.mode = mode
        self. model_type = model_type
        self.base = Path('../../../Logs')
        self.full_path = self.base / self.exp_id / self.mode / self.model_type
        self.loss_list = []
        os.makedirs(self.full_path)

    def log_summary(self, model, input_shape):
        sum_file = open(self.full_path / 'summary.txt', 'w+')
        sum_file.write(str(summary(model, input_data=input_shape, verbose=0)))
        sum_file.close()

    def log(self, loss_list):
        self.loss_list.append(loss_list)

    def done(self):
        pkl = open(self.full_path / 'loss.pkl', 'wb')
        pickle.dump(self.loss_list, pkl)
        pkl.close()
