import _init_paths
from trainer.teacher import Teacher


t = Teacher('new_test2')
t.add_siamese('resnet', '30', 1e-3, 32, 10)
t.do_exps()

# s = create_finetuner('fine', 'new_test1', 'densenet', 2, lr=5e-2, size=100)
# s.fit(10)
