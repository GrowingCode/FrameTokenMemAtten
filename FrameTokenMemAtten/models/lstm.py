from metas.hyper_settings import num_units, use_layer_norm
import tensorflow as tf
from utils.initializer import random_uniform_variable_initializer, \
  zero_variable_initializer, one_variable_initializer


class Y2DirectLSTMCell():
  
  def __init__(self, num_desc, forget_bias=0.0, activation=tf.nn.tanh):
    self.forget_bias = forget_bias
    self.activation = activation
    self.weights = tf.Variable(random_uniform_variable_initializer(22, 888+num_desc, [2 * num_units, 5 * num_units]))
    self.biases = tf.Variable(zero_variable_initializer([1, 5 * num_units]))
    if use_layer_norm:
      self.norm_weights = []
      self.norm_biases = []
      for _ in range(5):
        self.norm_weights.append(tf.Variable(one_variable_initializer([num_units])))
        self.norm_biases.append(tf.Variable(zero_variable_initializer([num_units])))
    
  def __call__(self, c1, h1, c2, h2):
    linear_input = tf.concat([h1, h2], 1)
    res = tf.matmul(linear_input, self.weights)
    res = tf.add(res, self.biases)
    i, j, f, f2, o = tf.split(value=res, num_or_size_splits=5, axis=1)
    if use_layer_norm:
      i = layer_normalization(i, self.norm_weights[0], self.norm_biases[0])
      j = layer_normalization(j, self.norm_weights[1], self.norm_biases[1])
      f = layer_normalization(f, self.norm_weights[2], self.norm_biases[2])
      f2 = layer_normalization(f2, self.norm_weights[3], self.norm_biases[3])
      o = layer_normalization(o, self.norm_weights[4], self.norm_biases[4])
    new_cell = (c1 * tf.nn.sigmoid(f + self.forget_bias) + 
                c2 * tf.nn.sigmoid(f2 + self.forget_bias) + 
             self.activation(j) * tf.nn.sigmoid(i))
    new_h = self.activation(new_cell) * tf.nn.sigmoid(o)
    return new_cell, new_h


class Y3DirectLSTMCell():
  
  def __init__(self, num_desc, forget_bias=0.0, activation=tf.nn.tanh):
    self.forget_bias = forget_bias
    self.activation = activation
    self.weights = tf.Variable(random_uniform_variable_initializer(111, 777+num_desc, [3 * num_units, 6 * num_units]))
    self.biases = tf.Variable(zero_variable_initializer([1, 6 * num_units]))
    if use_layer_norm:
      self.norm_weights = []
      self.norm_biases = []
      for _ in range(6):
        self.norm_weights.append(tf.Variable(one_variable_initializer([num_units])))
        self.norm_biases.append(tf.Variable(zero_variable_initializer([num_units])))
    
  def __call__(self, c1, h1, c2, h2, c3, h3):
    linear_input = tf.concat([h1, h2, h3], 1)
    res = tf.matmul(linear_input, self.weights)
    res = tf.add(res, self.biases)
    i, j, f, f2, f3, o = tf.split(value=res, num_or_size_splits=6, axis=1)
    if use_layer_norm:
      i = layer_normalization(i, self.norm_weights[0], self.norm_biases[0])
      j = layer_normalization(j, self.norm_weights[1], self.norm_biases[1])
      f = layer_normalization(f, self.norm_weights[2], self.norm_biases[2])
      f2 = layer_normalization(f2, self.norm_weights[3], self.norm_biases[3])
      f3 = layer_normalization(f3, self.norm_weights[4], self.norm_biases[4])
      o = layer_normalization(o, self.norm_weights[5], self.norm_biases[5])
    new_cell = (c1 * tf.nn.sigmoid(f + self.forget_bias) + 
                c2 * tf.nn.sigmoid(f2 + self.forget_bias) + 
                c3 * tf.nn.sigmoid(f3 + self.forget_bias) + 
             self.activation(j) * tf.nn.sigmoid(i))
#     new_cell = layer_normalization(new_cell, self.norm_weights[6], self.norm_biases[6])
    new_h = self.activation(new_cell) * tf.nn.sigmoid(o)
    return new_cell, new_h


class Y2DLSTMCell():
  
  def __init__(self, num_desc, forget_bias=0.0, activation=tf.nn.tanh):
    self.forget_bias = forget_bias
    self.activation = activation
    self.w = tf.Variable(random_uniform_variable_initializer(111, 777+num_desc, [3 * num_units, 5 * num_units]))
    self.b = tf.Variable(zero_variable_initializer([1, 5 * num_units]))
    if use_layer_norm:
      self.norm_weights = []
      self.norm_biases = []
      for _ in range(6):
        self.norm_weights.append(tf.Variable(one_variable_initializer([num_units])))
        self.norm_biases.append(tf.Variable(zero_variable_initializer([num_units])))
  
  def __call__(self, inputs, c1, h1, c2, h2):
    linear_input = tf.concat([inputs, h1, h2], 1)
    res = tf.matmul(linear_input, self.w)
    concat_ = tf.add(res, self.b)
    i, j, f, f2, o = tf.split(value=concat_, num_or_size_splits=5, axis=1)
    if use_layer_norm:
      i = layer_normalization(i, self.norm_weights[0], self.norm_biases[0])
      j = layer_normalization(j, self.norm_weights[1], self.norm_biases[1])
      f = layer_normalization(f, self.norm_weights[2], self.norm_biases[2])
      f2 = layer_normalization(f2, self.norm_weights[3], self.norm_biases[3])
      o = layer_normalization(o, self.norm_weights[4], self.norm_biases[4])
    new_c = c1 * tf.nn.sigmoid(f) + c2 * tf.nn.sigmoid(f2) + tf.nn.sigmoid(i) * tf.nn.tanh(j)
    new_h = self.activation(new_c) * tf.nn.sigmoid(o)
    return new_c, new_h


class YLSTMCell():
  
  def __init__(self, num_desc, forget_bias=0.0, activation=tf.nn.tanh):
    self.forget_bias = forget_bias
    self.activation = activation
    self.w = tf.Variable(random_uniform_variable_initializer(9, 88+num_desc, [2 * num_units, 4 * num_units]))
    self.b = tf.Variable(zero_variable_initializer([1, 4 * num_units]))
    if use_layer_norm:
      self.norm_weights = []
      self.norm_biases = []
      for _ in range(4):
        self.norm_weights.append(tf.Variable(one_variable_initializer([num_units])))
        self.norm_biases.append(tf.Variable(zero_variable_initializer([num_units])))
  
  def __call__(self, inputs, state):
    """
    Long short-term memory cell (LSTM)
    @param: inputs (batch,n)
    @param state: the states and hidden unit of the two cells
    """
    c, h = state
    linear_input = tf.concat([inputs, h], 1)
    res = tf.matmul(linear_input, self.w)
    res = tf.add(res, self.b)
    i, j, f, o = tf.split(value=res, num_or_size_splits=4, axis=1)
    if use_layer_norm:
      i = layer_normalization(i, self.norm_weights[0], self.norm_biases[0])
      j = layer_normalization(j, self.norm_weights[1], self.norm_biases[1])
      f = layer_normalization(f, self.norm_weights[2], self.norm_biases[2])
      o = layer_normalization(o, self.norm_weights[3], self.norm_biases[3])
    '''
    compute cell
    '''
    new_c1 = (c * tf.nn.sigmoid(f + self.forget_bias) + 
             self.activation(j) * tf.nn.sigmoid(i))
    '''
    compute h
    '''
    new_h1 = self.activation(new_c1) * tf.nn.sigmoid(o)
    return new_h1, (new_c1, new_h1)
    
  
def layer_normalization(need_to_normalize_tensor, scale, shift, epsilon=1e-5):
  """ Layer normalizes a 2D tensor along its second axis """
  m, v = tf.nn.moments(need_to_normalize_tensor, [1], keepdims=True)
  ln_initial = (need_to_normalize_tensor - m) / tf.sqrt(v + epsilon)
  return ln_initial * scale + shift

