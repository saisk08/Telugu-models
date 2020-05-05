import _init_paths
from trainer.teacher import Teacher


t = Teacher('new_test1')
t.add_siamese('all', 'all', 1e-3, 32, 10)
t.do_exps()
