import _init_paths
from trainer.teacher import Teacher, Fineteacher


'''
Run experiments with batch side of 32
Use MSE loss for Siamese
No similar images shown
'''
exp_name = 'exp4-batch32-mse'

t = Teacher(exp_name)
t.add_all('all', 'all', 1e-3, 32, 10)
t.do_exps()

s = Fineteacher(exp_name)
s.add_all('all', 'all', 5e-2, 32, 10)
s.do_exps()

s = Fineteacher('fine-batch32-mse', exp_name)
s.add_all('all', 'all', 1e-2, 32, 10)
s.do_exps()
