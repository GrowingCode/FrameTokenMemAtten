import tensorflow as tf
from metas.non_hyper_constants import float_type


variable_initializer_seed = 13

def random_uniform_variable_initializer(mt, bs, shape):
  return tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=float_type, seed=mt*variable_initializer_seed+bs)

def one_variable_initializer(shape):
  return tf.ones(shape, dtype=float_type)

def zero_variable_initializer(shape):
  return tf.zeros(shape, dtype=float_type)

