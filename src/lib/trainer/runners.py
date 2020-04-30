from trainer import create_trainer


def get_trainers(size, mode, exp_id, lr):
    return [create_trainer(exp_id, mode, 'normal', size=size, lr=lr),
            create_trainer(exp_id, mode, 'resent', size=size, lr=lr),
            create_trainer(exp_id, mode, 'dense', size=size)]


def sizes(sl, mode, exp_id, lr):
    return [get_trainers(x, mode, exp_id, lr) for x in sl]


class BaseTrainer():
    def __init__(self, exp_id):
        self.size_list = [30, 50, 100, 150, 200]
        self.exp_id = exp_id

    def do_experiments(self):
        raise NotImplementedError


class BatchTrainer(BaseTrainer):
    def __init__(self, lr, epoch_list, exp_id):
        super().__init__(self, exp_id)
        self.epoch_list = epoch_list

        for epoch in self.epoch_list:
            sup_trainers = sizes(self.size_list, 'supervised', self.exp_id, lr)
            sia_trainers = sizes(self.size_list, 'siamese', self.exp_id, lr)
            self.batch = [sup_trainers, sia_trainers]

    def do_experiments(self):
        for epoch in self.epoch_list:
            for trainers in self.batch:
                for t in trainers:
                    t.fit(epoch)


class SingleTrainer(BaseTrainer):
    def __init__(self, lr, epochs, exp_id):
        super().__init__(self, exp_id)
        self.epochs = epochs

        for epoch in self.epoch_list:
            sup_trainers = sizes(self.size_list, 'supervised', self.exp_id, lr)
            sia_trainers = sizes(self.size_list, 'siamese', self.exp_id, lr)
            self.batch = [sup_trainers, sia_trainers]

    def do_experiments(self):
        for trainers in self.batch:
            for t in trainers:
                t.fit(self.epochs)


class BasicTrainer(BaseTrainer):
    def __init__(self, lr, epochs, mode, model_type, exp_id):
        super().__init__(self, exp_id)
        self.epochs = epochs

        for epoch in self.epoch_list:
            self.batch = [create_trainer(self.exp_id, mode, model_type, size=x,
                                         lr=lr)
                          for x in self.size_list]

    def do_experiments(self):
        for t in self.batch:
            t.fit(self.epochs)
