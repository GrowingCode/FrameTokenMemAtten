import tensorflow as tf
from metas.non_hyper_constants import int_type


a = tf.ones([2,3], int_type)
b = a[-2:-1,:]
print(b)


