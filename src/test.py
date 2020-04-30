import _init_paths
from trainer.runners import BasicTrainer

t = BasicTrainer(1e-3, 10, 'supervised', 'resnet', 'test_01')
t.do_experiments()
