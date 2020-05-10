import tensorflow as tf
from metas.hyper_settings import num_units
from metas.non_hyper_constants import float_type, int_type
from models.lstm import YLSTMCell
from utils.initializer import random_uniform_variable_initializer
from models.embed_merger import EmbedMerger


class NTMOneDirection():
  
  def __init__(self, num_desc):
    self.merger = EmbedMerger(num_desc)
    self.memory_update_lstm = YLSTMCell(30+num_desc)
    self.initial_cell = tf.Variable(random_uniform_variable_initializer(175, 525+num_desc, [1, num_units]))
    self.initial_h = tf.Variable(random_uniform_variable_initializer(1755, 525+num_desc, [1, num_units]))
  
  def compute_variables_in_statement(self, var_info, token_info, forward_memory_cell, forward_memory_h, loop_forward_cell, loop_forward_h, loop_backward_cell, loop_backward_h):
    '''
    compute updated discrete embedding
    '''

    def updated_embed_loop_cond(u_start, u_end, *_):
      return tf.less_equal(u_start, u_end)
    
    def updated_embed_loop_body(u_start, u_end, discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h):
      token_type_content_en = token_info[u_start]
      local_token_id = var_info[u_start]
      discrete_memory_vars = tf.concat([discrete_memory_vars, [local_token_id]], axis=0)
      discrete_memory_tokens = tf.concat([discrete_memory_tokens, [token_type_content_en]], axis=0)
      f_index = u_start
      f_cell = [loop_forward_cell[f_index]]
      assert f_cell != None
      f_h = [loop_forward_h[f_index]]
      b_index = f_index
      b_cell = [loop_backward_cell[b_index]]
      assert b_cell != None
      b_h = [loop_backward_h[b_index]]
      one_mg = self.merger(f_h, b_h)
      ''' compute merged features '''
      local_token_valid = tf.cast(tf.logical_and(local_token_id > 0, local_token_id < tf.shape(forward_memory_h)[0]), int_type)
      real_id = local_token_valid * local_token_id
      m_cell = tf.stack([self.initial_cell, [forward_memory_cell[real_id]]])[local_token_valid]
      m_h = tf.stack([self.initial_h, [forward_memory_h[real_id]]])[local_token_valid]
      _, (m_forward_cell, m_forward_h) = self.memory_update_lstm(one_mg, (m_cell, m_h))
      discrete_forward_memory_cell = tf.concat([discrete_forward_memory_cell, m_forward_cell], axis=0)
      discrete_forward_memory_h = tf.concat([discrete_forward_memory_h, m_forward_h], axis=0)
      ''' slice if local_token_id is invalid '''
      slice_size = 1 - local_token_valid
      curr_size = tf.shape(discrete_memory_vars)[0]
      retain_size = curr_size - slice_size
      discrete_memory_vars = tf.slice(discrete_memory_vars, [0], [retain_size])
      discrete_memory_tokens = tf.slice(discrete_memory_tokens, [0], [retain_size])
      discrete_forward_memory_cell = tf.slice(discrete_forward_memory_cell, [0, 0], [retain_size, num_units])
      discrete_forward_memory_h = tf.slice(discrete_forward_memory_h, [0, 0], [retain_size, num_units])
      return u_start + 1, u_end, discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h
    
    discrete_memory_vars = tf.zeros([0], int_type)
    discrete_memory_tokens = tf.zeros([0], int_type)
    discrete_forward_memory_cell = tf.zeros([0, num_units], float_type)
    discrete_forward_memory_h = tf.zeros([0, num_units], float_type)
    _, _, discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h = tf.while_loop(updated_embed_loop_cond, updated_embed_loop_body, [0, tf.shape(var_info)[-1]-1, discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units])], parallel_iterations=1)
    return discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h
    
  def update_memory_with_variables_in_statement(self, memory_en, forward_memory_cell, forward_memory_h, stmt_var, stmt_en, stmt_forward_memory_cell, stmt_forward_memory_h):
    '''
    update the memory
    '''

    def update_memory_loop_cond(t, t_end, *_):
      return tf.less_equal(t, t_end)

    def update_memory_loop_body(t, t_end, updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h):
      updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h = update_one_variable(stmt_var[t], stmt_en[t], [stmt_forward_memory_cell[t]], [stmt_forward_memory_h[t]], updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h)
      return t + 1, t_end, updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h

    t, t_end = tf.constant(0, int_type), tf.shape(stmt_en)[0]-1
    updated_memory_tokens = memory_en
    updated_forward_memory_cell = forward_memory_cell
    updated_forward_memory_h = forward_memory_h
    _, _, updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h = tf.while_loop(update_memory_loop_cond, update_memory_loop_body, [t, t_end, updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units])], parallel_iterations=1)
    return updated_memory_tokens, updated_forward_memory_cell, updated_forward_memory_h
    
    
def update_one_variable(local_token_index, type_content_en, decode_dup_f_cell, decode_dup_f_h, dup_accumulated_en, dup_accumulated_cell, dup_accumulated_h):
  is_dup = local_token_index > 0
  is_dup_i = tf.cast(is_dup, int_type)
  before_part = [0, tf.stack([0, local_token_index])[is_dup_i]]
  curr_part = [0, tf.stack([0, 1])[is_dup_i]]
  after_part_start = tf.stack([0, local_token_index+1])[is_dup_i]
  after_part_length = tf.shape(dup_accumulated_en)[0]-after_part_start
  after_part_valid = tf.cast(after_part_length > 0, int_type)
  r_after_part_start = after_part_start * after_part_valid
  r_after_part_length = after_part_length * after_part_valid
  after_part = [r_after_part_start, r_after_part_length]
  ''' slice and concatenate en '''
#   a_op = tf.Assert(r_after_part_start >= 0, ["after_part_start", after_part_start, "r_after_part_start", r_after_part_start])
#   with tf.control_dependencies([a_op]):# , p_op
  before_part_dup_accumulated_en = tf.slice(dup_accumulated_en, [before_part[0]], [before_part[1]])
  curr_part_dup_accumulated_en = tf.slice(tf.expand_dims(type_content_en, 0), [curr_part[0]], [curr_part[1]])
  after_part_dup_accumulated_en = tf.slice(dup_accumulated_en, [after_part[0]], [after_part[1]])
  new_dup_accumulated_en = tf.concat([before_part_dup_accumulated_en, curr_part_dup_accumulated_en, after_part_dup_accumulated_en], axis=0)
  ''' slice and concatenate cell '''
  before_part_dup_accumulated_cell = tf.slice(dup_accumulated_cell, [before_part[0], 0], [before_part[1], num_units])
  curr_part_dup_accumulated_cell = tf.slice(decode_dup_f_cell, [curr_part[0], 0], [curr_part[1], num_units])
  after_part_dup_accumulated_cell = tf.slice(dup_accumulated_cell, [after_part[0], 0], [after_part[1], num_units])
  new_dup_accumulated_cell = tf.concat([before_part_dup_accumulated_cell, curr_part_dup_accumulated_cell, after_part_dup_accumulated_cell], axis=0)
  ''' slice and concatenate h '''
  before_part_dup_accumulated_h = tf.slice(dup_accumulated_h, [before_part[0], 0], [before_part[1], num_units])
  curr_part_dup_accumulated_h = tf.slice(decode_dup_f_h, [curr_part[0], 0], [curr_part[1], num_units])
  after_part_dup_accumulated_h = tf.slice(dup_accumulated_h, [after_part[0], 0], [after_part[1], num_units])
  new_dup_accumulated_h = tf.concat([before_part_dup_accumulated_h, curr_part_dup_accumulated_h, after_part_dup_accumulated_h], axis=0)
  return new_dup_accumulated_en, new_dup_accumulated_cell, new_dup_accumulated_h



