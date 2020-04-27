import tensorflow as tf
from models.lstm import layer_normalization
from utils.initializer import random_uniform_variable_initializer,\
  zero_variable_initializer, one_variable_initializer
from metas.hyper_settings import num_units, use_layer_norm,\
  use_lstm_merger_style


class EmbedMerger():
  
  def __init__(self):
    self.w = tf.Variable(random_uniform_variable_initializer(222, 333, [2 * num_units, 5 * num_units]))
    self.b = tf.Variable(zero_variable_initializer([1, 5 * num_units]))
    if use_layer_norm:
      self.norm_weights = []
      self.norm_biases = []
      for _ in range(6):
        self.norm_weights.append(tf.Variable(zero_variable_initializer([num_units])))
        self.norm_biases.append(tf.Variable(one_variable_initializer([num_units])))
  
  def __call__(self, forward_h, backward_h):
    linear_input = tf.concat([forward_h, backward_h], 1)
    res = tf.matmul(linear_input, self.w)
    res = tf.add(res, self.b)
    i, j, f, f2, o = tf.split(value=res, num_or_size_splits=5, axis=1)
    # add layer normalization to each gate
    if use_layer_norm:
      i = layer_normalization(i, self.norm_weights[0], self.norm_biases[0])
      j = layer_normalization(j, self.norm_weights[1], self.norm_biases[1])
      f = layer_normalization(f, self.norm_weights[2], self.norm_biases[2])
      f2 = layer_normalization(f2, self.norm_weights[3], self.norm_biases[3])
      o = layer_normalization(o, self.norm_weights[4], self.norm_biases[4])
    '''
    compute cell
    '''
    new_h = (forward_h * tf.nn.sigmoid(f) + backward_h * tf.nn.sigmoid(f2) + 
             tf.tanh(j) * tf.nn.sigmoid(i))
    if use_lstm_merger_style:
      if use_layer_norm:
        new_h = layer_normalization(new_h, self.norm_weights[5], self.norm_biases[5])
      new_h = self.activation(new_h) * tf.nn.sigmoid(o)
    return new_h
  
  