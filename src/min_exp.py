import _init_paths
from trainer.teacher import Teacher, Fineteacher


'''
Run experiments with batch side of 32
Use RD loss for Siamese
No similar images shown
'''
exp_name = 'exp1-batch32'

t = Teacher(exp_name)
t.add_all('dense', 'all', 1e-3, 32, 10)
t.do_exps()

s = Fineteacher(exp_name)
s.add_all('dense', 'all', 5e-2, 32, 10)
s.do_exps()

s = Fineteacher('fine1-batch32', exp_name)
s.add_all('dense', 'all', 1e-2, 32, 10)
s.do_exps()
