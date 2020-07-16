import numpy as np
from metas.non_hyper_constants import np_float_type


a = np.zeros([2], np_float_type)
b = np.ones([2], np_float_type)
c = np.ones([2], np_float_type)

a += b
a += c

print(a)


