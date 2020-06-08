import tensorflow as tf
from metas.hyper_settings import attention_algorithm, v_attention,\
  stand_attention, num_units, lstm_initialize_range
from metas.non_hyper_constants import float_type
from utils.initializer import random_uniform_variable_initializer


class YAttention():
  
  def __init__(self, num_desc):
    if attention_algorithm == v_attention:
      self.v = tf.Variable(random_uniform_variable_initializer(203, 21 + num_desc, [1, num_units]))
      self.w_v = tf.Variable(random_uniform_variable_initializer(2033, 231 + num_desc, [num_units, num_units], ini_range=lstm_initialize_range))
      self.w_h = tf.Variable(random_uniform_variable_initializer(2033, 131 + num_desc, [num_units, num_units], ini_range=lstm_initialize_range))
      self.w_ctx_c = tf.Variable(random_uniform_variable_initializer(203, 145 + num_desc, [2*num_units, num_units], ini_range=lstm_initialize_range))
    elif attention_algorithm == stand_attention:
      self.w = tf.Variable(random_uniform_variable_initializer(2033, 132 + num_desc, [num_units, num_units], ini_range=lstm_initialize_range))
    else:
      print("Strange Error! Unrecognized attention algorithm")
    self.w_ctx_h = tf.Variable(random_uniform_variable_initializer(203, 135 + num_desc, [2*num_units, num_units], ini_range=lstm_initialize_range))
      
  def compute_attention_logits(self, atten_hs, h):
    if attention_algorithm == v_attention:
      atten_size = tf.shape(atten_hs)[0]
      ons = tf.ones([atten_size], float_type)
      logits1 = tf.matmul(tf.matmul(self.v, self.w_v), atten_hs, transpose_b=True)
      logits1 = tf.squeeze(logits1, axis=0)
      logits2 = tf.matmul(self.v, tf.matmul(tf.matmul(self.w_h, h, transpose_b=True), [ons]))
      logits2 = tf.squeeze(logits2, axis=0)
      logits = logits1 + logits2
    elif attention_algorithm == stand_attention:
      logits = tf.matmul(tf.matmul(h, self.w), atten_hs, transpose_b=True)
      logits = tf.squeeze(logits, axis=0)
    else:
      print("Strange Error! Unrecognized attention algorithm")
    return logits
    
  def compute_attention_context(self, atten_hs, h):
    logits = self.compute_attention_logits(atten_hs, h)
    alpha = tf.nn.softmax(logits)
    c_t = tf.matmul([alpha], atten_hs)
    return c_t
  
  def compute_attention_h(self, atten_hs, h):
    c_t = self.compute_attention_context(atten_hs, h)
    linear_input = tf.concat([h, c_t], 1)
    c_h = tf.matmul(linear_input, self.w_ctx_h)
    return c_h
  
  def compute_attention_cell_h(self, atten_cells, cell, atten_hs, h):
    atten_size = tf.shape(atten_hs)[0]
    ons = tf.ones([atten_size], float_type)
    logits1 = tf.matmul(tf.matmul(self.v, self.w_v), atten_hs, transpose_b=True)
    logits1 = tf.squeeze(logits1, axis=0)
    logits2 = tf.matmul(self.v, tf.matmul(tf.matmul(self.w_h, h, transpose_b=True), [ons]))
    logits2 = tf.squeeze(logits2, axis=0)
    logits = logits1 + logits2
    alpha = tf.nn.softmax(logits)
    ctx_h = tf.matmul([alpha], atten_hs)
    linear_input_h = tf.concat([h, ctx_h], 1)
    ctx_h = tf.matmul(linear_input_h, self.w_ctx_h)
    ctx_c = tf.matmul([alpha], atten_cells)
    linear_input_c = tf.concat([cell, ctx_c], 1)
    ctx_c = tf.matmul(linear_input_c, self.w_ctx_c)
    return ctx_c, ctx_h
  
