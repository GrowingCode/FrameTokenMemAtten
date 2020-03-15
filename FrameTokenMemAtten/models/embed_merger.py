import tensorflow as tf
from models.lstm import layer_normalization
from utils.initializer import random_uniform_variable_initializer,\
  zero_variable_initializer, one_variable_initializer
from metas.hyper_settings import num_units


class EmbedMerger():
  
  def __init__(self):
    self.w = tf.Variable(random_uniform_variable_initializer(222, 333, [2 * num_units, 4 * num_units]))
    self.b = tf.Variable(zero_variable_initializer([1, 4 * num_units]))
    self.norm_wrights = []
    self.norm_biases = []
    for _ in range(4):
      self.norm_wrights.append(tf.Variable(zero_variable_initializer([num_units])))
      self.norm_biases.append(tf.Variable(one_variable_initializer([num_units])))
  
  def __call__(self, forward_h, backward_h):
    linear_input = tf.concat([forward_h, backward_h], 1)
    res = tf.matmul(linear_input, self.w)
    res = tf.add(res, self.b)
    i, j, f, o = tf.split(value=res, num_or_size_splits=4, axis=1)
    # add layer normalization to each gate
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
  
  