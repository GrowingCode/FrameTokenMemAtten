from metas.hyper_settings import initialize_range, initialize_seed_base
from metas.non_hyper_constants import float_type
import tensorflow as tf

variable_initializer_seed = 13

def random_uniform_variable_initializer(mt, bs, shape, ini_range=initialize_range):
  if ini_range == 0.0:
    return zero_variable_initializer(shape)
  return tf.random.uniform(shape, minval=-ini_range, maxval=ini_range, dtype=float_type, seed=mt*variable_initializer_seed+bs+initialize_seed_base)

def one_variable_initializer(shape):
  return tf.ones(shape, dtype=float_type)

def zero_variable_initializer(shape):
  return tf.zeros(shape, dtype=float_type)

