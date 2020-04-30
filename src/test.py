import _init_paths
from trainer.runners import BasicTrainer

t = BasicTrainer(1e-3, 10, 'siamese', 'resnet', 'test_03')
t.do_experiments()
