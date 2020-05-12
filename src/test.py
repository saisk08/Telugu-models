import _init_paths
from trainer.teacher import Freezeteacher


# t = Freezeteacher('new_test2')
# t.add_siamese('resnet', '30', 1e-3, 32, 10)
# t.do_exps()

# s = create_finetuner('fine', 'new_test1', 'densenet', 2, lr=5e-2, size=100)
# s.fit(10)

s = Freezeteacher('fine-test', 'new_test3')
s.add_all('resnet', 30, 1e-2, 32, 30)
s.do_exps()
