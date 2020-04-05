import tensorflow as tf
from metas.hyper_settings import top_ks, num_units, contingent_parameters_num,\
  use_dup_model, accumulated_token_max_length, compute_token_memory,\
  atom_decode_mode, token_decode, sword_decode, compose_tokens_of_a_statement,\
  token_embedder_mode, swords_compose_mode, token_only_mode, treat_first_element_as_skeleton,\
  take_lstm_states_as_memory_states
from utils.model_tensors_metrics import create_empty_tensorflow_tensors,\
  create_metrics_contingent_index, default_metrics_meta,\
  special_handle_metrics_meta
from utils.tensor_concat import concat_in_fixed_length_two_dimension,\
  concat_in_fixed_length_one_dimension
from models.lstm import YLSTMCell
from utils.initializer import random_uniform_variable_initializer
from inputs.atom_embeddings import AtomSimpleEmbed, BiLSTMEmbed,\
  sword_sequence_for_token
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings
from metas.non_hyper_constants import float_type, all_token_summary, SkeletonNum,\
  TokenNum, TotalNumberOfSubWord, int_type
from models.mem import NTMOneDirection
from models.dup_pattern import PointerNetwork
from models.token_sword_decode import decode_one_token,\
  decode_swords_of_one_token
from utils.tensor_array_stand import make_sure_shape_of_tensor_array
from models.embed_merger import EmbedMerger


class SkeletonDecodeModel():
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    self.type_content_data = type_content_data
    self.statistical_metrics_meta = default_metrics_meta + self.create_extra_default_metrics_meta() + special_handle_metrics_meta
    self.metrics_meta = self.statistical_metrics_meta + self.create_in_use_tensors_meta()
    self.metrics_name = [metric_m[0] for metric_m in self.metrics_meta]
    self.metrics_shape = [metric_m[1] for metric_m in self.metrics_meta]
    self.index_metrics = dict((k,v) for k, v in zip(range(len(self.metrics_name)), self.metrics_name))
    self.metrics_index = {value:key for key, value in self.index_metrics.items()}
    self.metrics_contingent_index = create_metrics_contingent_index(self.metrics_meta)
    self.contingent_parameters = tf.Variable(random_uniform_variable_initializer(2, 5, [contingent_parameters_num, 2, num_units]))
    self.skeleton_lstm_cell = YLSTMCell()
    self.skeleton_dup_lstm_cell = YLSTMCell()
    number_of_skeletons = self.type_content_data[all_token_summary][SkeletonNum] + 1
    self.one_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(258, 578, [number_of_skeletons, num_units]))
    self.linear_skeleton_output_w = tf.Variable(random_uniform_variable_initializer(257, 576, [number_of_skeletons, num_units]))
    self.one_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(258, 578, [number_of_skeletons, num_units]))
    self.one_dup_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(259, 579, [number_of_skeletons, num_units]))
    self.skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(255, 572, [number_of_skeletons, 2, num_units]))
    self.skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(252, 572, [number_of_skeletons, 2, num_units]))
    
    if compute_token_memory:
      self.mem_nn = NTMOneDirection()
      self.forward_token_lstm = YLSTMCell()
      self.backward_token_lstm = YLSTMCell()
      if compose_tokens_of_a_statement:
        self.tokens_merger = EmbedMerger()
        self.compose_lstm_cell = YLSTMCell()
    
    self.token_lstm = YLSTMCell()
    r_token_embedder_mode = token_embedder_mode
    number_of_tokens = self.type_content_data[all_token_summary][TokenNum] + 1
    number_of_subwords = self.type_content_data[all_token_summary][TotalNumberOfSubWord] + 1
    
    if atom_decode_mode == token_decode:
      self.linear_token_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_tokens, num_units]))
      
      self.dup_token_embedder, self.dup_token_lstm, self.token_pointer = None, None, None
      if use_dup_model:
        self.one_dup_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(252, 226, [number_of_tokens, num_units]))
        self.dup_token_embedder = AtomSimpleEmbed(self.one_dup_hot_token_embedding)
        self.dup_token_lstm = YLSTMCell()
        self.token_pointer = PointerNetwork()
        self.dup_skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(155, 572, [number_of_skeletons, 2, num_units]))
        self.dup_skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(152, 572, [number_of_skeletons, 2, num_units]))
        if compute_token_memory:
          self.dup_mem_nn = NTMOneDirection()
          self.dup_forward_token_lstm = YLSTMCell()
          self.dup_backward_token_lstm = YLSTMCell()
      
    elif atom_decode_mode == sword_decode:
      self.linear_sword_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_subwords, num_units]))
      self.sword_lstm = YLSTMCell()
      r_token_embedder_mode = swords_compose_mode
      
    else:
      assert False, "Wrong atom_decode_mode"
      
    if r_token_embedder_mode == token_only_mode:
      self.one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_tokens, num_units]))
      self.token_embedder = AtomSimpleEmbed(self.one_hot_token_embedding)
    elif r_token_embedder_mode == swords_compose_mode:
      self.one_hot_sword_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_subwords, num_units]))
      self.sword_embedder = AtomSimpleEmbed(self.one_hot_sword_embedding)
      self.token_embedder = BiLSTMEmbed(self.type_content_data, self.one_hot_sword_embedding)
    else:
      assert False, "Wrong token_embedder_mode"
      
      
  def create_extra_default_metrics_meta(self):
    return [("skeleton_loss", tf.TensorShape(())), ("skeleton_accurate", tf.TensorShape([len(top_ks)])), ("skeleton_mrr", tf.TensorShape(())), ("skeleton_count", tf.TensorShape(()))]
  
  def create_in_use_tensors_meta(self):
    result = [("token_accumulated_en", tf.TensorShape([None])), ("token_accumulated_cell", tf.TensorShape([None, num_units])), ("token_accumulated_h", tf.TensorShape([None, num_units])), ("dup_token_accumulated_cell", tf.TensorShape([None, num_units])), ("dup_token_accumulated_h", tf.TensorShape([None, num_units])), ("loop_forward_cells", tf.TensorShape([None, num_units])), ("loop_forward_hs", tf.TensorShape([None, num_units])), ("loop_backward_cells", tf.TensorShape([None, num_units])), ("loop_backward_hs", tf.TensorShape([None, num_units])), ("dup_loop_forward_cells", tf.TensorShape([None, num_units])), ("dup_loop_forward_hs", tf.TensorShape([None, num_units])), ("dup_loop_backward_cells", tf.TensorShape([None, num_units])), ("dup_loop_backward_hs", tf.TensorShape([None, num_units])), ("memory_accumulated_en", tf.TensorShape([None])), ("memory_cells", tf.TensorShape([None, num_units])), ("memory_hs", tf.TensorShape([None, num_units])), ("dup_memory_cells", tf.TensorShape([None, num_units])), ("dup_memory_hs", tf.TensorShape([None, num_units]))]
    return result
  
  def __call__(self, token_info_tensor, token_info_start_tensor, token_info_end_tensor, training = True):
    self.token_info_tensor = token_info_tensor
    self.token_info_start_tensor = token_info_start_tensor
    self.token_info_end_tensor = token_info_end_tensor
    self.training = training
    ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    f_res = tf.while_loop(self.stmt_iterate_cond, self.stmt_iterate_body, [0, tf.shape(self.token_info_start_tensor)[-1], *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2+len(self.statistical_metrics_meta)])
    f_res = list(post_process_decoder_output(f_res, self.metrics_index))
    return f_res
  
  def stmt_iterate_cond(self, i, i_len, *_):
    return tf.less(i, i_len)
  
  def stmt_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    
    stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
    
    stmt_start_offset = 0
    if treat_first_element_as_skeleton:
      stmt_start_offset = 1
      ''' handle skeleton '''
      skt_id = self.token_info_tensor[0][stmt_start]
      cell = tf.expand_dims(stmt_metrics[self.metrics_index["token_accumulated_cell"]][-1], 0)
      h = tf.expand_dims(stmt_metrics[self.metrics_index["token_accumulated_h"]][-1], 0)
      o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(self.training, self.linear_skeleton_output_w, skt_id, h)
      skt_id_valid = tf.cast(tf.greater(skt_id, 2), float_type)
      
      stmt_metrics[self.metrics_index["skeleton_loss"]] = stmt_metrics[self.metrics_index["skeleton_loss"]] + o_loss_of_this_node * skt_id_valid
      stmt_metrics[self.metrics_index["skeleton_accurate"]] = stmt_metrics[self.metrics_index["skeleton_accurate"]] + o_accurate_of_this_node * skt_id_valid
      stmt_metrics[self.metrics_index["skeleton_mrr"]] = stmt_metrics[self.metrics_index["skeleton_mrr"]] + o_mrr_of_this_node * skt_id_valid
      stmt_metrics[self.metrics_index["skeleton_count"]] = stmt_metrics[self.metrics_index["skeleton_count"]] + 1
      
      stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + o_loss_of_this_node * skt_id_valid
      stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + o_accurate_of_this_node * skt_id_valid
      stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + o_mrr_of_this_node * skt_id_valid
      stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + 1
      
      next_cell, next_h = self.skeleton_lstm_cell(tf.expand_dims(self.one_hot_skeleton_embedding[skt_id], 0), cell, h)
      stmt_metrics[self.metrics_index["token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["token_accumulated_cell"]], next_cell, accumulated_token_max_length)
      stmt_metrics[self.metrics_index["token_accumulated_h"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["token_accumulated_h"]], next_h, accumulated_token_max_length)
      
      stmt_metrics[self.metrics_index["token_accumulated_en"]] = concat_in_fixed_length_one_dimension(stmt_metrics[self.metrics_index["token_accumulated_en"]], [skt_id+50000000], accumulated_token_max_length)
      
      if use_dup_model:
        dup_cell = tf.expand_dims(stmt_metrics[self.metrics_index["dup_token_accumulated_cell"]][-1], 0)
        dup_h = tf.expand_dims(stmt_metrics[self.metrics_index["dup_token_accumulated_h"]][-1], 0)
        next_dup_cell, next_dup_h = self.skeleton_dup_lstm_cell(tf.expand_dims(self.one_dup_hot_skeleton_embedding[skt_id], 0), dup_cell, dup_h)
        stmt_metrics[self.metrics_index["dup_token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_token_accumulated_cell"]], next_dup_cell, accumulated_token_max_length)
        stmt_metrics[self.metrics_index["dup_token_accumulated_h"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_token_accumulated_h"]], next_dup_h, accumulated_token_max_length)
    
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
    return (ctn_start+1, ctn_end, i, *res_stmt_metrics)
  
  def itearate_tokens(self, i, stmt_metrics):
    
    b_stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
    
    stmt_start_offset = 0
    if treat_first_element_as_skeleton:
      stmt_start_offset = 1
      skt_id = self.token_info_tensor[0][b_stmt_start]
    else:
      stmt_start_offset = 0
      skt_id = 0
    
    stmt_start = b_stmt_start + stmt_start_offset
    
    ini_cells_hs = []
    '''
    iterate tokens
    '''
    ini_f_cell = tf.expand_dims(self.skeleton_forward_cell_h[skt_id][0], 0)
    ini_f_h = tf.expand_dims(self.skeleton_forward_cell_h[skt_id][1], 0)
    ini_b_cell = tf.expand_dims(self.skeleton_backward_cell_h[skt_id][0], 0)
    ini_b_h = tf.expand_dims(self.skeleton_backward_cell_h[skt_id][1], 0)
    ini_cells_hs.append(ini_f_cell)
    ini_cells_hs.append(ini_f_h)
    ini_cells_hs.append(ini_b_cell)
    ini_cells_hs.append(ini_b_h)
    
    if use_dup_model:
      dup_ini_f_cell = tf.expand_dims(self.dup_skeleton_forward_cell_h[skt_id][0], 0)
      dup_ini_f_h = tf.expand_dims(self.dup_skeleton_forward_cell_h[skt_id][1], 0)
      dup_ini_b_cell =  tf.expand_dims(self.dup_skeleton_backward_cell_h[skt_id][0], 0)
      dup_ini_b_h = tf.expand_dims(self.dup_skeleton_backward_cell_h[skt_id][1], 0)
      ini_cells_hs.append(dup_ini_f_cell)
      ini_cells_hs.append(dup_ini_f_h)
      ini_cells_hs.append(dup_ini_b_cell)
      ini_cells_hs.append(dup_ini_b_h)
      
    ini_f_cell = ini_cells_hs[0]
    ini_f_h = ini_cells_hs[1]
    ini_b_cell = ini_cells_hs[2]
    ini_b_h = ini_cells_hs[3]
    if use_dup_model:
      dup_ini_f_cell = ini_cells_hs[4]
      dup_ini_f_h = ini_cells_hs[5]
      dup_ini_b_cell = ini_cells_hs[6]
      dup_ini_b_h = ini_cells_hs[7]
    '''
    leaf info also means variable info
    '''
    info_length = stmt_end-stmt_start+1
    leaf_info = tf.slice(self.token_info_tensor[1], [stmt_start], [info_length])
    token_info = tf.slice(self.token_info_tensor[0], [stmt_start], [info_length])
    
    if compose_tokens_of_a_statement:
      begin_cell = tf.expand_dims(stmt_metrics[self.metrics_index["token_accumulated_cell"]][-1], 0)
      begin_h = tf.expand_dims(stmt_metrics[self.metrics_index["token_accumulated_h"]][-1], 0)
    
    f_res = tf.while_loop(self.token_iterate_cond, self.token_iterate_body, [stmt_start, stmt_end, stmt_start, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = list(f_res[3:])
    
    if compute_token_memory:
      '''
      compute token memory and compute repetition
      '''
      if take_lstm_states_as_memory_states:
        discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h = leaf_info, token_info, stmt_metrics[self.metrics_index["token_accumulated_cell"]][-info_length:, :], stmt_metrics[self.metrics_index["token_accumulated_h"]][-info_length:, :]
        if use_dup_model:
          dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h = leaf_info, token_info, stmt_metrics[self.metrics_index["dup_token_accumulated_cell"]][-info_length:, :], stmt_metrics[self.metrics_index["dup_token_accumulated_h"]][-info_length:, :]
      else:
        embeds, dup_embeds = tf.zeros([0, num_units], float_type), tf.zeros([0, num_units], float_type)
        _, _, embeds, dup_embeds = tf.while_loop(self.token_embed_cond, self.token_embed_body, [stmt_start, stmt_end, embeds, dup_embeds], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units])], parallel_iterations=1)
        
        stmt_metrics[self.metrics_index["loop_forward_cells"]] = ini_f_cell
        stmt_metrics[self.metrics_index["loop_forward_hs"]] = ini_f_h
        if use_dup_model:
          stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = dup_ini_f_cell
          stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = dup_ini_f_h
        f_res = tf.while_loop(self.forward_loop_cond, self.forward_loop_body, [0, stmt_end-(stmt_start), embeds, dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
        stmt_metrics = list(f_res[4:])
        stmt_metrics[self.metrics_index["loop_backward_cells"]] = ini_b_cell
        stmt_metrics[self.metrics_index["loop_backward_hs"]] = ini_b_h
        if use_dup_model:
          stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = dup_ini_b_cell
          stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = dup_ini_b_h
        f_res = tf.while_loop(self.backward_loop_cond, self.backward_loop_body, [0, stmt_end-(stmt_start), embeds, dup_embeds, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), *self.metrics_shape], parallel_iterations=1)
        stmt_metrics = list(f_res[4:])
        
        discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h = self.mem_nn.compute_variables_in_statement(leaf_info, token_info, stmt_metrics[self.metrics_index["memory_cells"]], stmt_metrics[self.metrics_index["memory_hs"]], stmt_metrics[self.metrics_index["loop_forward_cells"]], stmt_metrics[self.metrics_index["loop_forward_hs"]], stmt_metrics[self.metrics_index["loop_backward_cells"]], stmt_metrics[self.metrics_index["loop_backward_hs"]])
        if use_dup_model:
          dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h = self.dup_mem_nn.compute_variables_in_statement(leaf_info, token_info, stmt_metrics[self.metrics_index["dup_memory_cells"]], stmt_metrics[self.metrics_index["dup_memory_hs"]], stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], stmt_metrics[self.metrics_index["dup_loop_backward_hs"]])
      
      mem_acc_en = stmt_metrics[self.metrics_index["memory_accumulated_en"]]
      updated_memory_accumulated_en, stmt_metrics[self.metrics_index["memory_cells"]], stmt_metrics[self.metrics_index["memory_hs"]] = self.mem_nn.update_memory_with_variables_in_statement(mem_acc_en, stmt_metrics[self.metrics_index["memory_cells"]], stmt_metrics[self.metrics_index["memory_hs"]], discrete_memory_vars, discrete_memory_tokens, discrete_forward_memory_cell, discrete_forward_memory_h)
      stmt_metrics[self.metrics_index["memory_accumulated_en"]] = updated_memory_accumulated_en
      if use_dup_model:
        _, stmt_metrics[self.metrics_index["dup_memory_cells"]], stmt_metrics[self.metrics_index["dup_memory_hs"]] = self.dup_mem_nn.update_memory_with_variables_in_statement(mem_acc_en, stmt_metrics[self.metrics_index["dup_memory_cells"]], stmt_metrics[self.metrics_index["dup_memory_hs"]], dup_discrete_memory_vars, dup_discrete_memory_tokens, dup_discrete_forward_memory_cell, dup_discrete_forward_memory_h)
      
      if compose_tokens_of_a_statement:
        '''
        compute BiLSTM composition of tokens of a statement
        '''
        merged_tokens_embed = self.tokens_merger([stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]], [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]])
        end_cell, end_h = self.compose_lstm_cell(merged_tokens_embed, begin_cell, begin_h)
        stmt_metrics[self.metrics_index["token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["token_accumulated_cell"]], end_cell, accumulated_token_max_length)
        stmt_metrics[self.metrics_index["token_accumulated_h"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["token_accumulated_h"]], end_h, accumulated_token_max_length)
      
    return stmt_metrics
  
  def token_iterate_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_iterate_body(self, i, i_len, ini_i, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
    oracle_type_content_var = self.token_info_tensor[1][i]
    oracle_type_content_var_relative = self.token_info_tensor[2][i]
    if atom_decode_mode == token_decode:
      r_stmt_metrics_tuple = decode_one_token(self.training, oracle_type_content_en, oracle_type_content_var, oracle_type_content_var_relative, self.metrics_index, stmt_metrics, self.linear_token_output_w, self.token_lstm, self.token_embedder, self.dup_token_lstm, self.dup_token_embedder, self.token_pointer)
    elif atom_decode_mode == sword_decode:
      oracle_sword_en_sequence = sword_sequence_for_token(self.type_content_data, oracle_type_content_en)
      r_stmt_metrics_tuple = decode_swords_of_one_token(self.training, oracle_type_content_en, oracle_sword_en_sequence, self.metrics_index, self.metrics_shape, stmt_metrics, self.token_lstm, self.token_embedder, self.linear_sword_output_w, self.sword_embedder, self.sword_lstm)
    else:
      assert False
    return (i + 1, i_len, ini_i, *r_stmt_metrics_tuple)
  
  '''
  build memory
  '''
  def token_embed_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_embed_body(self, i, i_len, embeds, dup_embeds):
    oracle_type_content_en = self.token_info_tensor[0][i]
    e_emebd = self.token_embedder.compute_h(oracle_type_content_en)
    embeds = tf.concat([embeds, e_emebd], axis=0)
    if use_dup_model:
      dup_e_emebd = self.dup_token_embedder.compute_h(oracle_type_content_en)
      dup_embeds = tf.concat([dup_embeds, dup_e_emebd], axis=0)
    return (i+1, i_len, embeds, dup_embeds)
 
  def forward_loop_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def forward_loop_body(self, i, i_len, embeds, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    f_cell = [stmt_metrics[self.metrics_index["loop_forward_cells"]][-1]]
    f_h = [stmt_metrics[self.metrics_index["loop_forward_hs"]][-1]]
    new_f_cell, new_f_h = self.forward_token_lstm([embeds[i]], f_cell, f_h)
    stmt_metrics[self.metrics_index["loop_forward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["loop_forward_cells"]], new_f_cell, accumulated_token_max_length)
    stmt_metrics[self.metrics_index["loop_forward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["loop_forward_hs"]], new_f_h, accumulated_token_max_length)
    if use_dup_model:
      dup_f_cell = [stmt_metrics[self.metrics_index["dup_loop_forward_cells"]][0]]
      dup_f_h = [stmt_metrics[self.metrics_index["dup_loop_forward_hs"]][0]]
      new_dup_f_cell, new_dup_f_h = self.dup_forward_token_lstm([dup_embeds[i]], dup_f_cell, dup_f_h)
      stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], new_dup_f_cell, accumulated_token_max_length)
      stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], new_dup_f_h, accumulated_token_max_length)
    return (i+1, i_len, embeds, dup_embeds, *stmt_metrics)
  
  def backward_loop_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def backward_loop_body(self, i, i_len, embeds, dup_embeds, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    b_cell = [stmt_metrics[self.metrics_index["loop_backward_cells"]][0]]
    b_h = [stmt_metrics[self.metrics_index["loop_backward_hs"]][0]]
    new_b_cell, new_b_h = self.backward_token_lstm([embeds[i_len]], b_cell, b_h)
    stmt_metrics[self.metrics_index["loop_backward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["loop_backward_cells"]], new_b_cell, accumulated_token_max_length)
    stmt_metrics[self.metrics_index["loop_backward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["loop_backward_hs"]], new_b_h, accumulated_token_max_length)
    if use_dup_model:
      dup_b_cell = [stmt_metrics[self.metrics_index["dup_loop_backward_cells"]][0]]
      dup_b_h = [stmt_metrics[self.metrics_index["dup_loop_backward_hs"]][0]]
      new_dup_b_cell, new_dup_b_h = self.dup_backward_token_lstm([dup_embeds[i_len]], dup_b_cell, dup_b_h)
      stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], new_dup_b_cell, accumulated_token_max_length)
      stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_backward_hs"]], new_dup_b_h, accumulated_token_max_length)
    return (i, i_len-1, embeds, dup_embeds, *stmt_metrics)


def post_process_decoder_output(model_metrics, metrics_index):
  t_array = model_metrics[metrics_index["atom_beam"]]
  model_metrics[metrics_index["atom_beam"]] = make_sure_shape_of_tensor_array(t_array)
  return model_metrics






