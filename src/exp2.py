import _init_paths
from trainer.teacher import Teacher, Fineteacher


'''
Run experiments with batch side of 64
Use RD loss for Siamese
No similar images shown
'''
exp_name = 'exp2-batch64'

t = Teacher(exp_name)
t.add_all('all', 'all', 1e-3, 64, 10)
t.do_exps()

s = Fineteacher(exp_name)
s.add_all('all', 'all', 1e-2, 64, 10)
s.do_exps()

exp_name = 'exp3-batch64-lr'

t = Teacher(exp_name)
t.add_all('all', 'all', 5e-3, 64, 10)
t.do_exps()

s = Fineteacher(exp_name)
s.add_all('all', 'all', 5e-2, 64, 10)
s.do_exps()
