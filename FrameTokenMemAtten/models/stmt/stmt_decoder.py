from metas.hyper_settings import compute_token_memory,\
  compose_tokens_of_a_statement, compose_mode, compose_one_way_lstm,\
  compose_one_way_lstm_mode, one_way_three_way_compose, one_way_two_way_compose,\
  compose_bi_way_lstm, num_units, decode_attention_way, decode_no_attention,\
  token_embedder_mode, atom_decode_mode, token_decode,\
  token_only_mode, compose_half_one_way_lstm,\
  one_way_stand_compose
from models.mem import NTMOneDirection
from models.lstm import YLSTMCell, Y2DirectLSTMCell, Y3DirectLSTMCell
from models.embed_merger import EmbedMerger
from utils.initializer import random_uniform_variable_initializer
import tensorflow as tf
from models.attention import YAttention
from metas.non_hyper_constants import all_token_summary, TokenHitNum,\
  float_type, int_type
from models.token_sword_decode import TokenDecoder
from inputs.atom_embeddings import TokenAtomEmbed
from models.lstm_procedure import backward_varied_lstm_steps, one_lstm_step
from utils.tensor_concat import concat_in_fixed_length_two_dimension
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from models.basic_decoder import BasicDecodeModel


class StatementDecodeModel(BasicDecodeModel):
  
  def __init__(self, type_content_data):
    super(StatementDecodeModel, self).__init__(type_content_data)
    
    self.ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    
    self.skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(255, 572, [1, 2, num_units]))
    self.skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(252, 572, [1, 2, num_units]))
    
    if compute_token_memory:
      self.mem_nn = NTMOneDirection(500)
      self.forward_token_lstm = YLSTMCell(3)
      self.backward_token_lstm = YLSTMCell(4)
      if compose_tokens_of_a_statement:
        if compose_mode == compose_one_way_lstm and (compose_one_way_lstm_mode == one_way_two_way_compose or compose_one_way_lstm_mode == one_way_three_way_compose):
          self.tokens_merger = Y2DirectLSTMCell(150)
        else:
          self.tokens_merger = EmbedMerger(150)
        if compose_mode == compose_one_way_lstm and compose_one_way_lstm_mode == one_way_three_way_compose:
          self.compose_lstm_cell = Y3DirectLSTMCell(5)
        else:
          self.compose_lstm_cell = YLSTMCell(5)
        if compose_mode == compose_bi_way_lstm:
          self.bi_way_merger = Y2DirectLSTMCell(155)
          self.bi_way_ini_cell = tf.Variable(random_uniform_variable_initializer(855, 55, [1, num_units]))
          self.bi_way_ini_h = tf.Variable(random_uniform_variable_initializer(850, 50, [1, num_units]))
          self.bi_way_lstm = YLSTMCell(52)
     
    self.token_attention = None
    if decode_attention_way > decode_no_attention:
      self.token_attention = YAttention(10)
     
    self.token_lstm = YLSTMCell(0)
    r_token_embedder_mode = token_embedder_mode
     
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    # number_of_subwords = self.type_content_data[all_token_summary][SwordHitNum]
     
    if atom_decode_mode == token_decode:
      self.linear_token_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_tokens, num_units]))
      self.dup_token_embedder, self.dup_token_lstm, self.dup_token_pointer = None, None, None
#       if use_dup_model:
#         self.one_dup_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(252, 226, [number_of_tokens, num_units]))
#         self.dup_token_embedder = TokenAtomEmbed(self.type_content_data, self.one_dup_hot_token_embedding)
#         self.dup_token_lstm = YLSTMCell(9)
#         self.dup_token_pointer = PointerNetwork(655)
#         self.dup_skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(155, 572, [number_of_skeletons, 2, num_units]))
#         self.dup_skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(152, 572, [number_of_skeletons, 2, num_units]))
#         if compute_token_memory:
#           self.dup_mem_nn = NTMOneDirection(800)
#           self.dup_forward_token_lstm = YLSTMCell(10)
#           self.dup_backward_token_lstm = YLSTMCell(11)
      self.token_decoder = TokenDecoder(type_content_data, self.metrics_index, self.linear_token_output_w, self.token_attention, self.dup_token_pointer)
       
    # elif atom_decode_mode == sword_decode:
    #   self.linear_sword_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_subwords, num_units]))
    #   self.sword_lstm = YLSTMCell(12)
    #   r_token_embedder_mode = swords_compose_mode
      
    else:
      assert False, "Wrong atom_decode_mode"
       
    if r_token_embedder_mode == token_only_mode:
      self.one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_tokens, num_units]))
      self.token_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_embedding)
    # elif r_token_embedder_mode == swords_compose_mode:
    #   self.one_hot_sword_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_subwords, num_units]))
    #   self.sword_embedder = SwordAtomEmbed(self.type_content_data, self.one_hot_sword_embedding)
    #   self.token_embedder = BiLSTMEmbed(self.type_content_data, self.one_hot_sword_embedding)
    else:
      assert False, "Wrong token_embedder_mode"
  
  def create_in_use_tensors_meta(self):
    result = super(StatementDecodeModel, self).create_in_use_tensors_meta() + [("loop_forward_cells", tf.TensorShape([None, num_units])), ("loop_forward_hs", tf.TensorShape([None, num_units])), ("loop_backward_cells", tf.TensorShape([None, num_units])), ("loop_backward_hs", tf.TensorShape([None, num_units])), ("dup_loop_forward_cells", tf.TensorShape([None, num_units])), ("dup_loop_forward_hs", tf.TensorShape([None, num_units])), ("dup_loop_backward_cells", tf.TensorShape([None, num_units])), ("dup_loop_backward_hs", tf.TensorShape([None, num_units])), ("f_stmt_cell", tf.TensorShape([1, num_units])), ("f_stmt_h", tf.TensorShape([1, num_units]))]
    return result
  
  def set_up_field_when_calling(self, one_example, training):
    self.token_info_tensor = one_example[0]
    self.token_info_start_tensor = one_example[1]
    self.token_info_end_tensor = one_example[2]
    self.training = training
  
  def __call__(self, one_example, training = True):
    self.set_up_field_when_calling(one_example, training);
    f_res = tf.while_loop(self.stmt_iterate_cond, self.stmt_iterate_body, [0, tf.shape(self.token_info_start_tensor)[-1], *self.ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2+len(self.statistical_metrics_meta)])
#     f_res = list(post_process_decoder_output(f_res, self.metrics_index))
#     if print_accurate_of_each_example:
#       p_op = tf.print(["accurate:", f_res[self.metrics_index["all_accurate"]], "count:", f_res[self.metrics_index["all_count"]]])
#       with tf.control_dependencies([p_op]):
#         f_res[self.metrics_index["all_count"]] += 0
    return f_res
  
  def stmt_iterate_cond(self, i, i_len, *_):
    return tf.less(i, i_len)
  
  def stmt_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
     
    stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
     
    '''
    this step ignores the statement with no type content tokens (only with skeleton token). 
    '''
    itearate_tokens_continue = tf.cast(stmt_end >= stmt_start, int_type)
    f_res = tf.while_loop(self.itearate_tokens_cond, self.itearate_tokens_body, [tf.constant(0, int_type), itearate_tokens_continue, stmt_start, stmt_end, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = f_res[4:]
    return (i + 1, i_len, *stmt_metrics)
  
  def itearate_tokens_cond(self, ctn_start, ctn_end, *_):
    return tf.less(ctn_start, ctn_end)
  
  def itearate_tokens_body(self, ctn_start, ctn_end, stmt_start, stmt_end, *stmt_metrics):
    res_stmt_metrics = self.itearate_tokens(stmt_start, stmt_end, stmt_metrics)
    return (ctn_start+1, ctn_end, stmt_start, stmt_end, *res_stmt_metrics)
  
  def itearate_tokens(self, stmt_start, stmt_end, stmt_metrics):
    
#     b_stmt_start = self.token_info_start_tensor[i]
#     stmt_end = self.token_info_end_tensor[i]
    
#     stmt_start_offset = 0
    
#     if self.treat_first_element_as_skeleton:
#       stmt_start_offset = 1
#       skt_id = self.token_info_tensor[0][b_stmt_start]
#       skt_id_valid_bool = tf.logical_and(tf.greater(skt_id, 2), tf.less(skt_id, self.type_content_data[all_token_summary][SkeletonHitNum]))
#       skt_use_id = tf.stack([UNK_en, skt_id])[tf.cast(skt_id_valid_bool, int_type)]
#     else:
#       stmt_start_offset = 0
    skt_use_id = 0
    
#     stmt_start = b_stmt_start + stmt_start_offset
     
    '''
    iterate tokens
    '''
    ini_f_cell = tf.expand_dims(self.skeleton_forward_cell_h[skt_use_id][0], 0)
    ini_f_h = tf.expand_dims(self.skeleton_forward_cell_h[skt_use_id][1], 0)
    ini_b_cell = tf.expand_dims(self.skeleton_backward_cell_h[skt_use_id][0], 0)
    ini_b_h = tf.expand_dims(self.skeleton_backward_cell_h[skt_use_id][1], 0)
    
#     if use_dup_model:
#       dup_ini_f_cell = tf.expand_dims(self.dup_skeleton_forward_cell_h[skt_use_id][0], 0)
#       dup_ini_f_h = tf.expand_dims(self.dup_skeleton_forward_cell_h[skt_use_id][1], 0)
#       dup_ini_b_cell =  tf.expand_dims(self.dup_skeleton_backward_cell_h[skt_use_id][0], 0)
#       dup_ini_b_h = tf.expand_dims(self.dup_skeleton_backward_cell_h[skt_use_id][1], 0)
      
#     '''
#     leaf info also means variable info
#     '''
#     info_length = stmt_end-stmt_start+1
#     token_info = tf.slice(self.token_info_tensor[0], [stmt_start], [info_length])
#     leaf_info = tf.slice(self.token_info_tensor[1], [stmt_start], [info_length])
    
    f_res = tf.while_loop(self.token_iterate_cond, self.token_iterate_body, [stmt_start, stmt_end, stmt_start, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = list(f_res[3:])
     
    if compute_token_memory:
      '''
      compute token memory and compute repetition
      '''
      embeds = tf.zeros([0, num_units], float_type)
      dup_embeds = tf.zeros([0, num_units], float_type)
#       , dup_embeds, tf.TensorShape([None, num_units])
      _, _, embeds = tf.while_loop(self.token_embed_cond, self.token_embed_body, [stmt_start, stmt_end, embeds], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units])], parallel_iterations=1)
       
      stmt_metrics[self.metrics_index["loop_forward_cells"]] = ini_f_cell
      stmt_metrics[self.metrics_index["loop_forward_hs"]] = ini_f_h
#       if use_dup_model:
#         stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = dup_ini_f_cell
#         stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = dup_ini_f_h
      f_res = tf.while_loop(self.forward_loop_cond, self.forward_loop_body, [0, stmt_end-(stmt_start), embeds, dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
      stmt_metrics = list(f_res[4:])
      stmt_metrics[self.metrics_index["loop_backward_cells"]] = ini_b_cell
      stmt_metrics[self.metrics_index["loop_backward_hs"]] = ini_b_h
#       if use_dup_model:
#         stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = dup_ini_b_cell
#         stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = dup_ini_b_h
      f_res = tf.while_loop(self.backward_loop_cond, self.backward_loop_body, [0, stmt_end-(stmt_start), embeds, dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
      stmt_metrics = list(f_res[4:])
      
#       discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h = self.mem_nn.compute_variables_in_statement(leaf_info, token_info, stmt_metrics[self.metrics_index["memory_acc_cell"]], stmt_metrics[self.metrics_index["memory_acc_h"]], stmt_metrics[self.metrics_index["loop_forward_cells"]], stmt_metrics[self.metrics_index["loop_forward_hs"]], stmt_metrics[self.metrics_index["loop_backward_cells"]], stmt_metrics[self.metrics_index["loop_backward_hs"]])
#       if use_dup_model:
#         dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h = self.dup_mem_nn.compute_variables_in_statement(leaf_info, token_info, stmt_metrics[self.metrics_index["dup_memory_acc_cell"]], stmt_metrics[self.metrics_index["dup_memory_acc_h"]], stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], stmt_metrics[self.metrics_index["dup_loop_backward_hs"]])
      
#       stmt_metrics[self.metrics_index["memory_en"]], stmt_metrics[self.metrics_index["memory_acc_cell"]], stmt_metrics[self.metrics_index["memory_acc_h"]] = self.mem_nn.update_memory_with_variables_in_statement(stmt_metrics[self.metrics_index["memory_en"]], stmt_metrics[self.metrics_index["memory_acc_cell"]], stmt_metrics[self.metrics_index["memory_acc_h"]], discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h)
#       if use_dup_model:
#         p_op = tf.print(["dup_discrete_memory_vars:", dup_discrete_memory_vars, "dup_discrete_memory_tokens:", dup_discrete_memory_tokens], summarize=100)
#         with tf.control_dependencies([p_op]):
#         stmt_metrics[self.metrics_index["dup_memory_en"]], stmt_metrics[self.metrics_index["dup_memory_acc_cell"]], stmt_metrics[self.metrics_index["dup_memory_acc_h"]] = self.dup_mem_nn.update_memory_with_variables_in_statement(stmt_metrics[self.metrics_index["dup_memory_en"]], stmt_metrics[self.metrics_index["dup_memory_acc_cell"]], stmt_metrics[self.metrics_index["dup_memory_acc_h"]], dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h)
        
#       if use_dup_model:
#         assert token_memory_mode == only_memory_mode and not compose_tokens_of_a_statement
        
      if compose_tokens_of_a_statement:
        '''
        compute BiLSTM composition of tokens of a statement
        '''
        f_stmt_cell, f_stmt_h = stmt_metrics[self.metrics_index["f_stmt_cell"]], stmt_metrics[self.metrics_index["f_stmt_h"]]
        if compose_mode == compose_half_one_way_lstm:
          merged_tokens_embed = tf.expand_dims(stmt_metrics[self.metrics_index["loop_forward_hs"]][-1], axis=0)
        elif compose_mode == compose_one_way_lstm:
          if compose_one_way_lstm_mode == one_way_stand_compose:
            merged_tokens_embed = self.tokens_merger([stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]], [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]])
          elif compose_one_way_lstm_mode == one_way_two_way_compose or compose_one_way_lstm_mode == one_way_three_way_compose:
            _, merged_tokens_embed = self.tokens_merger([stmt_metrics[self.metrics_index["loop_forward_cells"]][-1]], [stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]], [stmt_metrics[self.metrics_index["loop_backward_cells"]][0]], [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]])
          else:
            assert False
        elif compose_mode == compose_bi_way_lstm:
          merged_tokens_embed = self.tokens_merger([stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]], [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]])
        else:
          assert False
        stmt_metrics[self.metrics_index["dup_memory_acc_h"]] = tf.concat([stmt_metrics[self.metrics_index["dup_memory_acc_h"]], merged_tokens_embed], axis=0)
        
        if compose_mode == compose_half_one_way_lstm:
          _, (f_stmt_cell, f_stmt_h) = self.compose_lstm_cell(merged_tokens_embed, (f_stmt_cell, f_stmt_h))
        elif compose_mode == compose_one_way_lstm:
          if compose_one_way_lstm_mode == one_way_stand_compose or compose_one_way_lstm_mode == one_way_two_way_compose:
            _, (f_stmt_cell, f_stmt_h) = self.compose_lstm_cell(merged_tokens_embed, (f_stmt_cell, f_stmt_h))
          elif compose_one_way_lstm_mode == one_way_three_way_compose:
            f_stmt_cell, f_stmt_h = self.compose_lstm_cell(f_stmt_cell, f_stmt_h, [stmt_metrics[self.metrics_index["loop_forward_cells"]][-1]], [stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]], [stmt_metrics[self.metrics_index["loop_backward_cells"]][0]], [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]])
          else:
            assert False
        elif compose_mode == compose_bi_way_lstm:
          _, (f_stmt_cell, f_stmt_h) = self.compose_lstm_cell(merged_tokens_embed, (f_stmt_cell, f_stmt_h))
        else:
          assert False
        
        if compose_mode == compose_one_way_lstm or compose_mode == compose_half_one_way_lstm:
          for_token_cell, for_token_h = f_stmt_cell, f_stmt_h
        elif compose_mode == compose_bi_way_lstm:
          _, (b_stmt_cell, b_stmt_h) = backward_varied_lstm_steps(stmt_metrics[self.metrics_index["dup_memory_acc_h"]], self.bi_way_ini_cell, self.bi_way_ini_h, self.bi_way_lstm)
          for_token_cell, for_token_h = self.bi_way_merger(b_stmt_cell, b_stmt_h, f_stmt_cell, f_stmt_h)
        else:
          assert False
        stmt_metrics[self.metrics_index["token_cell"]], stmt_metrics[self.metrics_index["token_h"]] = for_token_cell, for_token_h
        stmt_metrics[self.metrics_index["f_stmt_cell"]], stmt_metrics[self.metrics_index["f_stmt_h"]] = f_stmt_cell, f_stmt_h
       
    return stmt_metrics
  
  def token_iterate_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_iterate_body(self, i, i_len, ini_i, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
    oracle_type_content_var = self.token_info_tensor[1][i]
    oracle_type_content_var_relative = self.token_info_tensor[2][i]
#     conserved_memory_length = self.token_info_tensor[3][i]
    token_kind = self.token_info_tensor[4][i]
    if atom_decode_mode == token_decode:
      stmt_metrics = self.token_decoder.decode_one_token(stmt_metrics, self.training, oracle_type_content_en, oracle_type_content_var, oracle_type_content_var_relative, token_kind)
      stmt_metrics = one_lstm_step("", stmt_metrics, self.metrics_index, oracle_type_content_en, self.token_lstm, self.token_embedder)
#       if use_dup_model:
#         if compute_token_memory:
#           stmt_metrics = one_lstm_step("dup_", stmt_metrics, self.metrics_index, oracle_type_content_en, self.dup_token_lstm, self.dup_token_embedder)
#         else:
#           stmt_metrics = one_lstm_step_and_update_memory("dup_", stmt_metrics, self.metrics_index, oracle_type_content_en, oracle_type_content_var, conserved_memory_length, self.dup_token_lstm, self.dup_token_embedder)
#     elif atom_decode_mode == sword_decode:
#       oracle_sword_en_sequence = sword_sequence_for_token(self.type_content_data, oracle_type_content_en)
#       r_stmt_metrics_tuple = decode_swords_of_one_token(self.type_content_data, self.training, oracle_type_content_en, oracle_sword_en_sequence, self.metrics_index, self.metrics_shape, stmt_metrics, self.token_lstm, self.token_embedder, self.linear_sword_output_w, self.sword_embedder, self.sword_lstm)
    else:
      assert False
    return (i + 1, i_len, ini_i, *stmt_metrics)
  
  '''
  build memory
  '''
  def token_embed_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_embed_body(self, i, i_len, embeds):
    oracle_type_content_en = self.token_info_tensor[0][i]
    e_emebd = self.token_embedder.compute_h(oracle_type_content_en)
    embeds = tf.concat([embeds, e_emebd], axis=0)
#     if use_dup_model:
#       dup_e_emebd = self.dup_token_embedder.compute_h(oracle_type_content_en)
#       dup_embeds = tf.concat([dup_embeds, dup_e_emebd], axis=0)
    return (i+1, i_len, embeds)
  
  def forward_loop_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def forward_loop_body(self, i, i_len, embeds, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    f_cell = [stmt_metrics[self.metrics_index["loop_forward_cells"]][-1]]
    f_h = [stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]]
    _, (new_f_cell, new_f_h) = self.forward_token_lstm(tf.expand_dims(embeds[i], axis=0), (f_cell, f_h))
    stmt_metrics[self.metrics_index["loop_forward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["loop_forward_cells"]], new_f_cell, -1)
    stmt_metrics[self.metrics_index["loop_forward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["loop_forward_hs"]], new_f_h, -1)
#     if use_dup_model:
#       dup_f_cell = [stmt_metrics[self.metrics_index["dup_loop_forward_cells"]][-1]]
#       dup_f_h = [stmt_metrics[self.metrics_index["dup_loop_forward_hs"]][-1]]
#       _, (new_dup_f_cell, new_dup_f_h) = self.dup_forward_token_lstm(tf.expand_dims(dup_embeds[i], axis=0), (dup_f_cell, dup_f_h))
#       stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], new_dup_f_cell, -1)
#       stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], new_dup_f_h, -1)
    return (i+1, i_len, embeds, dup_embeds, *stmt_metrics)
  
  def backward_loop_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def backward_loop_body(self, i, i_len, embeds, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    b_cell = [stmt_metrics[self.metrics_index["loop_backward_cells"]][0]]
    b_h = [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]]
    _, (new_b_cell, new_b_h) = self.backward_token_lstm(tf.expand_dims(embeds[i_len], axis=0), (b_cell, b_h))
    stmt_metrics[self.metrics_index["loop_backward_cells"]] = concat_in_fixed_length_two_dimension(new_b_cell, stmt_metrics[self.metrics_index["loop_backward_cells"]], -1)
    stmt_metrics[self.metrics_index["loop_backward_hs"]] = concat_in_fixed_length_two_dimension(new_b_h, stmt_metrics[self.metrics_index["loop_backward_hs"]], -1)
#     if use_dup_model:
#       dup_b_cell = [stmt_metrics[self.metrics_index["dup_loop_backward_cells"]][0]]
#       dup_b_h = [stmt_metrics[self.metrics_index["dup_loop_backward_hs"]][0]]
#       _, (new_dup_b_cell, new_dup_b_h) = self.dup_backward_token_lstm(tf.expand_dims(dup_embeds[i_len], axis=0), (dup_b_cell, dup_b_h))
#       stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = concat_in_fixed_length_two_dimension(new_dup_b_cell, stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], -1)
#       stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = concat_in_fixed_length_two_dimension(new_dup_b_h, stmt_metrics[self.metrics_index["dup_loop_backward_hs"]], -1)
    return (i, i_len-1, embeds, dup_embeds, *stmt_metrics)
  
  
# def post_process_decoder_output(model_metrics, metrics_index):
#   t_array = model_metrics[metrics_index["atom_beam_noavg"]]
#   model_metrics[metrics_index["atom_beam_noavg"]] = make_sure_shape_of_tensor_array(t_array)
#   return model_metrics
 
 
 
  
  




