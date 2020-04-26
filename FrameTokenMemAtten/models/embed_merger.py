import tensorflow as tf
from models.lstm import layer_normalization
from utils.initializer import random_uniform_variable_initializer, \
  zero_variable_initializer, one_variable_initializer
from metas.hyper_settings import num_units, use_layer_norm
from metas.non_hyper_constants import float_type, learning_scope


class EmbedMerger():
  
  def __init__(self, num_desc):
    with tf.variable_scope(learning_scope):
      self.w = tf.get_variable("em_w" + str(num_desc), shape=[2 * num_units, 4 * num_units], dtype=float_type, initializer=random_uniform_variable_initializer(222, 333 + num_desc))
      self.b = tf.get_variable("em_b" + str(num_desc), shape=[1, 4 * num_units], dtype=float_type, initializer=zero_variable_initializer())
      if use_layer_norm:
        self.norm_wrights = []
        self.norm_biases = []
        for i in range(4):
          self.norm_weights.append(tf.get_variable("em_nw" + str(i) + str(num_desc), shape=[num_units], dtype=float_type, one_variable_initializer()))
          self.norm_biases.append(tf.get_variable("em_nb" + str(i) + str(num_desc), shape=[num_units], dtype=float_type, zero_variable_initializer()))
  
  def __call__(self, forward_h, backward_h):
    linear_input = tf.concat([forward_h, backward_h], 1)
    res = tf.matmul(linear_input, self.w)
    res = tf.add(res, self.b)
    i, j, f, o = tf.split(value=res, num_or_size_splits=4, axis=1)
    # add layer normalization to each gate
    if use_layer_norm:
      i = layer_normalization(i, self.norm_wrights[0], self.norm_biases[0])
      j = layer_normalization(j, self.norm_wrights[1], self.norm_biases[1])
      f = layer_normalization(f, self.norm_wrights[2], self.norm_biases[2])
      o = layer_normalization(o, self.norm_wrights[3], self.norm_biases[3])
    '''
    compute cell
    '''
    new_h = (forward_h * tf.nn.sigmoid(f) + backward_h * tf.nn.sigmoid(o) + 
             tf.tanh(j) * tf.nn.sigmoid(i))
    return new_h
  
  
