from .trainer import create_trainer


class Teacher():
    def __init__(self, exp_id):
        self.exp_id = exp_id
        self.exps = []
        self.data_size = [30, 50, 100, 150, 200]

    def get_types(self, model_type):
        if model_type == 'all':
            return ['resnet', 'normal', 'dense']
        else:
            return [model_type]

    def get_sizes(self, sizes):
        if sizes == 'all':
            return self.data_size
        elif isinstance(sizes, list):
            return sizes
        else:
            return [sizes]

    def add_experiment(self, mode, model_type, size, lr, bs, epochs):
        if mode == 'supervised':
            exp = {'trainer': create_trainer(
                self.exp_id, mode, model_type, size=size, lr=lr, bs=bs),
                'epochs': epochs}
            self.exps.append(exp)
            return
        exp1 = {'trainer': create_trainer(
                self.exp_id, mode, model_type, size=size, lr=lr, bs=bs, version=1),
                'epochs': epochs}
        exp2 = {'trainer': create_trainer(
                self.exp_id, mode, model_type, size=size, lr=lr, bs=bs, version=2),
                'epochs': epochs}
        self.append(exp1)
        self.append(exp2)

    def add_supervised(self, model_type, size, lr, bs):
        for mt in self.get_types(model_type):
            for s in self.get_sizes(size):
                self.add_experiment('supervised', mt, s, lr, bs, epochs)

    def add_siamese(info_list):
        for mt in self.get_types(model_type):
            for s in self.get_sizes(size):
                self.add_experiment('siamese', mt, s, lr, bs, epochs)

    def do_exps(self):
        for e in self.exps:
            e['trainer'].fit(e['epochs'])
