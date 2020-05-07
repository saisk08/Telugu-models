import _init_paths
from trainer.teacher import Teacher
from trainer.trainer import create_finetuner


# t = Teacher('new_test1')
# t.add_siamese('all', 'all', 1e-3, 32, 10)
# t.do_exps()

s = create_finetuner('fine', 'new_test1', 'resnet', 2, lr=5e-2, size=100)
s.fit(10)
