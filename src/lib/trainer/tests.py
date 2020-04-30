import torch


class Tester():
    def __init__(self, model, test_dl, logger, metric):
        self.model = model
        self.test_dl = test_dl
        self.logger = logger
        self.metric = metric
        self.model.to(self.device)

    def fit(self, epochs):
        self.model.eval()
        with torch.no_grad():
            tot_acc = 0.
            for xb, yb in self.test_dl:
                pred = self.model(xb)
                tot_acc += self.metric(pred, yb)
        nv = len(self.test_dl)
        acc = tot_acc / nv

        print('Acc: {}'.format(acc))
        self.logger.add_results(acc)
