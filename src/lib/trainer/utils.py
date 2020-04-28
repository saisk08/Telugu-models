from trainer import create_trainer


def get_trainers(size, mode):
    return [create_trainer(mode, 'normal', size=size),
            create_trainer(mode, 'resent', size=size),
            create_trainer(mode, 'dense', size=size)]


def sizes(sl, mode):
    return [get_trainers(x, mode) for x in sl]


class BaseTrainer():
    def __init__(self):
        self.size_list = [30, 50, 100, 150, 200]

    def do_experiments(self, exp_id):
        raise NotImplementedError


class BatchTrainer(BaseTrainer):
    def __init__(self, lr, epoch_list):
        super().__init__()
        self.epoch_list = epoch_list

        for epoch in self.epoch_list:
            sup_trainers = sizes(self.size_list, 'supervised')
            sia_trainers = sizes(self.size_list, 'siamese')
            self.batch = [sup_trainers, sia_trainers]

    def do_experiments(self, exp_id):
        for epoch in self.epoch_list:
            for trainers in self.batch:
                for t in trainers:
                    t.fit(epoch)


class SingleTrainer(BaseTrainer):
    def __init__(self, lr, epochs):
        super().__init__()
        self.epochs = epochs

        for epoch in self.epoch_list:
            sup_trainers = sizes(self.size_list, 'supervised')
            sia_trainers = sizes(self.size_list, 'siamese')
            self.batch = [sup_trainers, sia_trainers]

    def do_experiments(self, exp_id):
        for trainers in self.batch:
            for t in trainers:
                t.fit(self.epochs)


class BasicTrainer(BaseTrainer):
    def __init__(self, lr, epochs, mode, model_type):
        super().__init__()
        self.epochs = epochs

        for epoch in self.epoch_list:
            self.batch = [create_trainer(mode, model_type, size=x)
                          for x in self.size_list]

    def do_experiments(self, exp_id):
        for t in self.batch:
            t.fit(self.epochs)
