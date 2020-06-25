from inputs.atom_embeddings import TokenAtomEmbed, SkeletonAtomEmbed
from metas.hyper_settings import num_units, \
  compute_token_memory, \
  atom_decode_mode, token_decode, \
  only_memory_mode, token_memory_mode, \
  print_accurate_of_each_example
from metas.non_hyper_constants import float_type, all_token_summary, \
  int_type, SkeletonHitNum, TokenHitNum, UNK_en, skeleton_base
from models.basic_decoder import BasicDecodeModel
from models.dup_pattern import PointerNetwork
from models.lstm import YLSTMCell
from models.lstm_procedure import one_lstm_step, one_lstm_step_and_update_memory
from models.mem import NTMOneDirection
from models.token_sword_decode import DupTokenDecoder
import tensorflow as tf
from utils.initializer import random_uniform_variable_initializer
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from utils.tensor_concat import concat_in_fixed_length_two_dimension


class SkeletonDupModel(BasicDecodeModel):
  
  def __init__(self, type_content_data):
    super(SkeletonDupModel, self).__init__(type_content_data)
    
    self.treat_first_element_as_skeleton = 1
    
    number_of_skeletons = self.type_content_data[all_token_summary][SkeletonHitNum]
     
    if self.treat_first_element_as_skeleton:
      self.skeleton_dup_lstm_cell = YLSTMCell(2)
      self.one_dup_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(259, 579, [number_of_skeletons, num_units]))
      self.dup_skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_dup_hot_skeleton_embedding)
     
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    
    if atom_decode_mode == token_decode:
      self.one_dup_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(252, 226, [number_of_tokens, num_units]))
      self.dup_token_embedder = TokenAtomEmbed(self.type_content_data, self.one_dup_hot_token_embedding)
      self.dup_token_lstm = YLSTMCell(9)
      self.dup_token_pointer = PointerNetwork(655)
      self.dup_skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(155, 572, [number_of_skeletons, 2, num_units]))
      self.dup_skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(152, 572, [number_of_skeletons, 2, num_units]))
      if compute_token_memory:
        self.dup_mem_nn = NTMOneDirection(800)
        self.dup_forward_token_lstm = YLSTMCell(10)
        self.dup_backward_token_lstm = YLSTMCell(11)
      self.dup_token_decoder = DupTokenDecoder(type_content_data, self.metrics_index, self.dup_token_pointer) 
    else:
      assert False, "Wrong atom_decode_mode"
    
  def create_in_use_tensors_meta(self):
    result = super(SkeletonDupModel, self).create_in_use_tensors_meta() + [("dup_loop_forward_cells", tf.TensorShape([None, num_units])), ("dup_loop_forward_hs", tf.TensorShape([None, num_units])), ("dup_loop_backward_cells", tf.TensorShape([None, num_units])), ("dup_loop_backward_hs", tf.TensorShape([None, num_units]))]
    return result
  
  def __call__(self, one_example, training=True):
    self.token_info_tensor = one_example[0]
    self.token_info_start_tensor = one_example[1]
    self.token_info_end_tensor = one_example[2]
    self.token_base_model_accuracy = one_example[3]
    self.token_base_model_mrr = one_example[4]
    self.training = training
    ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    f_res = tf.while_loop(self.stmt_iterate_cond, self.stmt_iterate_body, [0, tf.shape(self.token_info_start_tensor)[-1], *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2 + len(self.statistical_metrics_meta)])
    if print_accurate_of_each_example:
      p_op = tf.print(["accurate:", f_res[self.metrics_index["all_accurate"]], "count:", f_res[self.metrics_index["all_count"]]])
      with tf.control_dependencies([p_op]):
        f_res[self.metrics_index["all_count"]] += 0
    return f_res
  
  def stmt_iterate_cond(self, i, i_len, *_):
    return tf.less(i, i_len)
  
  def stmt_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
     
    stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
     
    stmt_start_offset = 0
    if self.treat_first_element_as_skeleton:
      stmt_start_offset = 1
      ''' handle skeleton '''
      skt_id = self.token_info_tensor[0][stmt_start] - skeleton_base
       
      dup_cell = stmt_metrics[self.metrics_index["dup_token_cell"]]
      dup_h = stmt_metrics[self.metrics_index["dup_token_h"]]
      dup_skt_embed = self.dup_skeleton_embedder.compute_h(skt_id)
      _, (next_dup_cell, next_dup_h) = self.skeleton_dup_lstm_cell(dup_skt_embed, (dup_cell, dup_h))
      stmt_metrics[self.metrics_index["dup_token_cell"]] = next_dup_cell
      stmt_metrics[self.metrics_index["dup_token_h"]] = next_dup_h
    else:
      stmt_start_offset = 0
    
    '''
    this step ignores the statement with no type content tokens (only with skeleton token). 
    '''
    r_stmt_start = stmt_start + stmt_start_offset
    itearate_tokens_continue = tf.cast(stmt_end >= r_stmt_start, int_type)
    f_res = tf.while_loop(self.itearate_tokens_cond, self.itearate_tokens_body, [tf.constant(0, int_type), itearate_tokens_continue, i, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = f_res[3:]
    return (i + 1, i_len, *stmt_metrics)
  
  def itearate_tokens_cond(self, ctn_start, ctn_end, *_):
    return tf.less(ctn_start, ctn_end)
  
  def itearate_tokens_body(self, ctn_start, ctn_end, i, *stmt_metrics):
    res_stmt_metrics = self.itearate_tokens(i, stmt_metrics)
    return (ctn_start + 1, ctn_end, i, *res_stmt_metrics)
  
  def itearate_tokens(self, i, stmt_metrics):
    
    b_stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
    
    stmt_start_offset = 0
    if self.treat_first_element_as_skeleton:
      stmt_start_offset = 1
      skt_id = self.token_info_tensor[0][b_stmt_start]
      skt_id_valid_bool = tf.logical_and(tf.greater(skt_id, 2), tf.less(skt_id, self.type_content_data[all_token_summary][SkeletonHitNum]))
      skt_use_id = tf.stack([UNK_en, skt_id])[tf.cast(skt_id_valid_bool, int_type)]
    else:
      stmt_start_offset = 0
      skt_use_id = 0
    
    stmt_start = b_stmt_start + stmt_start_offset
     
    '''
    iterate tokens
    '''
    dup_ini_f_cell = tf.expand_dims(self.dup_skeleton_forward_cell_h[skt_use_id][0], 0)
    dup_ini_f_h = tf.expand_dims(self.dup_skeleton_forward_cell_h[skt_use_id][1], 0)
    dup_ini_b_cell = tf.expand_dims(self.dup_skeleton_backward_cell_h[skt_use_id][0], 0)
    dup_ini_b_h = tf.expand_dims(self.dup_skeleton_backward_cell_h[skt_use_id][1], 0)
      
    '''
    leaf info also means variable info
    '''
    info_length = stmt_end - stmt_start + 1
    token_info = tf.slice(self.token_info_tensor[0], [stmt_start], [info_length])
    leaf_info = tf.slice(self.token_info_tensor[1], [stmt_start], [info_length])
    
    f_res = tf.while_loop(self.token_iterate_cond, self.token_iterate_body, [stmt_start, stmt_end, stmt_start, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = list(f_res[3:])
     
    if compute_token_memory:
      '''
      compute token memory and compute repetition
      '''
      dup_embeds = tf.zeros([0, num_units], float_type)
      e_res = tf.while_loop(self.token_embed_cond, self.token_embed_body, [stmt_start, stmt_end, dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
      dup_embeds = e_res[2]
      
      stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = dup_ini_f_cell
      stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = dup_ini_f_h
      f_res = tf.while_loop(self.forward_loop_cond, self.forward_loop_body, [0, stmt_end - (stmt_start), dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
      stmt_metrics = list(f_res[3:])
      stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = dup_ini_b_cell
      stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = dup_ini_b_h
      f_res = tf.while_loop(self.backward_loop_cond, self.backward_loop_body, [0, stmt_end - (stmt_start), dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
      stmt_metrics = list(f_res[3:])
      
      dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h = self.dup_mem_nn.compute_variables_in_statement(leaf_info, token_info, stmt_metrics[self.metrics_index["dup_memory_acc_cell"]], stmt_metrics[self.metrics_index["dup_memory_acc_h"]], stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], stmt_metrics[self.metrics_index["dup_loop_backward_hs"]])
      stmt_metrics[self.metrics_index["dup_memory_en"]], stmt_metrics[self.metrics_index["dup_memory_acc_cell"]], stmt_metrics[self.metrics_index["dup_memory_acc_h"]] = self.dup_mem_nn.update_memory_with_variables_in_statement(stmt_metrics[self.metrics_index["dup_memory_en"]], stmt_metrics[self.metrics_index["dup_memory_acc_cell"]], stmt_metrics[self.metrics_index["dup_memory_acc_h"]], dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h)
        
      assert token_memory_mode == only_memory_mode
       
    return stmt_metrics
  
  def token_iterate_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_iterate_body(self, i, i_len, ini_i, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
    oracle_type_content_var = self.token_info_tensor[1][i]
    oracle_type_content_var_relative = self.token_info_tensor[2][i]
    conserved_memory_length = self.token_info_tensor[3][i]
    token_kind = self.token_info_tensor[4][i]
    base_model_accuracy = self.token_base_model_accuracy[i]
    base_model_mrr = self.token_base_model_mrr[i]
    if atom_decode_mode == token_decode:
      stmt_metrics = self.dup_token_decoder.decode_one_token(stmt_metrics, self.training, oracle_type_content_en, oracle_type_content_var, oracle_type_content_var_relative, token_kind, base_model_accuracy, base_model_mrr)
      if compute_token_memory:
        stmt_metrics = one_lstm_step("dup_", stmt_metrics, self.metrics_index, oracle_type_content_en, self.dup_token_lstm, self.dup_token_embedder)
      else:
        stmt_metrics = one_lstm_step_and_update_memory("dup_", stmt_metrics, self.metrics_index, oracle_type_content_en, oracle_type_content_var, conserved_memory_length, self.dup_token_lstm, self.dup_token_embedder)
    else:
      assert False
    return (i + 1, i_len, ini_i, *stmt_metrics)
  
  '''
  build memory
  '''

  def token_embed_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_embed_body(self, i, i_len, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
    dup_e_emebd = self.dup_token_embedder.compute_h(oracle_type_content_en)
    if compute_token_memory:
      oracle_type_content_var = self.token_info_tensor[1][i]
      mem_hs = stmt_metrics[self.metrics_index["dup_memory_acc_h"]]
      use_mem = tf.cast(tf.logical_and(tf.greater(oracle_type_content_var, 0), tf.less(oracle_type_content_var, tf.shape(mem_hs)[0])), int_type)
      r_var = oracle_type_content_var * use_mem
      dup_e_emebd = tf.stack([dup_e_emebd, [mem_hs[r_var]]])[use_mem]
    dup_embeds = tf.concat([dup_embeds, dup_e_emebd], axis=0)
    return (i + 1, i_len, dup_embeds, *stmt_metrics_tuple)
  
  def forward_loop_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def forward_loop_body(self, i, i_len, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    dup_f_cell = [stmt_metrics[self.metrics_index["dup_loop_forward_cells"]][-1]]
    dup_f_h = [stmt_metrics[self.metrics_index["dup_loop_forward_hs"]][-1]]
    _, (new_dup_f_cell, new_dup_f_h) = self.dup_forward_token_lstm(tf.expand_dims(dup_embeds[i], axis=0), (dup_f_cell, dup_f_h))
    stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], new_dup_f_cell, -1)
    stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], new_dup_f_h, -1)
    return (i + 1, i_len, dup_embeds, *stmt_metrics)
  
  def backward_loop_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def backward_loop_body(self, i, i_len, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    dup_b_cell = [stmt_metrics[self.metrics_index["dup_loop_backward_cells"]][0]]
    dup_b_h = [stmt_metrics[self.metrics_index["dup_loop_backward_hs"]][0]]
    _, (new_dup_b_cell, new_dup_b_h) = self.dup_backward_token_lstm(tf.expand_dims(dup_embeds[i_len], axis=0), (dup_b_cell, dup_b_h))
    stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = concat_in_fixed_length_two_dimension(new_dup_b_cell, stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], -1)
    stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = concat_in_fixed_length_two_dimension(new_dup_b_h, stmt_metrics[self.metrics_index["dup_loop_backward_hs"]], -1)
    return (i, i_len - 1, dup_embeds, *stmt_metrics)






