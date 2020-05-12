import _init_paths
from trainer.teacher import Freezeteacher

s = Freezeteacher('freeze-batch32-mse', 'exp4-batch32-mse')
s.add_all('all', 'all', 1e-2, 32, 10)
s.do_exps()

s = Freezeteacher('freeze-batch32-l1', 'exp3-batch32-l1')
s.add_all('all', 'all', 1e-2, 32, 10)
s.do_exps()

s = Freezeteacher('fine-batch32-rd', 'exp1-batch32')
s.add_all('all', 'all', 1e-2, 32, 10)
s.do_exps()

s = Freezeteacher('freeze1-batch32-mse', 'exp4-batch32-mse')
s.add_all('all', 'all', 5e-2, 32, 10)
s.do_exps()

s = Freezeteacher('freeze1-batch32-l1', 'exp3-batch32-l1')
s.add_all('all', 'all', 5e-2, 32, 10)
s.do_exps()

s = Freezeteacher('fine1-batch32-rd', 'exp1-batch32')
s.add_all('all', 'all', 5e-2, 32, 10)
s.do_exps()
