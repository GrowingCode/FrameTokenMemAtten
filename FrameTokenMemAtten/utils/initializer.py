import tensorflow as tf
from metas.non_hyper_constants import float_type, initialize_range,\
  initialize_seed_base


variable_initializer_seed = 13

def random_uniform_variable_initializer(mt, bs, shape=None):
  assert shape == None
  return tf.random_uniform_initializer(-initialize_range, initialize_range, seed=mt*variable_initializer_seed+bs+initialize_seed_base, dtype=float_type)
#   return tf.random.uniform(shape, minval=-initialize_range, maxval=initialize_range, dtype=float_type, seed=mt*variable_initializer_seed+bs+initialize_seed_base)

def one_variable_initializer():
  return tf.constant_initializer(1.0, dtype=float_type)

def zero_variable_initializer():
  return tf.constant_initializer(0.0, dtype=float_type)


