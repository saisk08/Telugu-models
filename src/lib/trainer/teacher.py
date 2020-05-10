from .trainer import create_trainer
from .trainer import create_finetuner


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
            exp = {'mode': mode, 'model_type': model_type, 'size': size,
                   'lr': lr, 'bs': bs,
                   'epochs': epochs, 'version': None}
            self.exps.append(exp)
            return
        exp1 = {'mode': mode,
                'model_type': model_type, 'size': size, 'lr': lr, 'bs': bs,
                'version': 1,  'epochs': epochs}
        exp2 = {'mode': mode,
                'model_type': model_type, 'size': size, 'lr': lr, 'bs': bs,
                'version': 2,  'epochs': epochs}
        self.exps.append(exp1)
        self.exps.append(exp2)

    def add_supervised(self, model_type, size, lr, bs, epochs):
        for mt in self.get_types(model_type):
            for s in self.get_sizes(size):
                self.add_experiment('supervised', mt, s, lr, bs, epochs)

    def add_siamese(self, model_type, size, lr, bs, epochs):
        for mt in self.get_types(model_type):
            for s in self.get_sizes(size):
                self.add_experiment('siamese', mt, s, lr, bs, epochs)

    def add_all(self, model_type, size, lr, bs, epochs):
        self.add_siamese(model_type, size, lr, bs, epochs)
        self.add_supervised(model_type, size, lr, bs, epochs)

    def do_exps(self):
        for e in self.exps:
            t = create_trainer(self.exp_id, e['mode'],
                               e['model_type'], lr=e['lr'], bs=e['bs'],
                               size=e['size'], version=e['version'])
            t.fit(e['epochs'])


class Fineteacher():
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

    def add_experiment(self, model_type, size, lr, bs, epochs):
        exp1 = {
            'model_type': model_type, 'size': size, 'lr': lr, 'bs': bs,
            'version': 1,  'epochs': epochs}
        exp2 = {
            'model_type': model_type, 'size': size, 'lr': lr, 'bs': bs,
            'version': 2,  'epochs': epochs}
        self.exps.append(exp1)
        self.exps.append(exp2)

    def add_all(self, model_type, size, lr, bs, epochs):
        for mt in self.get_types(model_type):
            for s in self.get_sizes(size):
                self.add_experiment(mt, s, lr, bs, epochs)

    def do_exps(self):
        for e in self.exps:
            t = create_finetuner(self.exp_id, e['model_type'], lr=e['lr'],
                                 bs=e['bs'], size=e['size'],
                                 version=e['version'])
            t.fit(e['epochs'])
