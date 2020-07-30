from inputs.atom_embeddings import BiLSTMEmbed, \
  TokenAtomEmbed, SwordAtomEmbed, SkeletonAtomEmbed
from metas.hyper_settings import num_units, compute_token_memory, \
  atom_decode_mode, token_decode, sword_decode, compose_tokens_of_a_statement, \
  token_embedder_mode, swords_compose_mode, token_only_mode, \
  decode_attention_way, decode_no_attention, compose_one_way_lstm, compose_mode,\
  compose_bi_way_lstm, compose_half_one_way_lstm, compose_one_way_lstm_mode,\
  one_way_stand_compose, one_way_two_way_compose, one_way_three_way_compose,\
  print_accurate_of_each_example
from metas.non_hyper_constants import float_type, all_token_summary, \
  int_type, SkeletonHitNum, SwordHitNum, TokenHitNum, UNK_en
from models.attention import YAttention
from models.basic_decoder import BasicDecodeModel
from models.embed_merger import EmbedMerger
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings
from models.lstm import YLSTMCell, Y2DirectLSTMCell, Y3DirectLSTMCell
from models.mem import NTMOneDirection
import tensorflow as tf
from utils.initializer import random_uniform_variable_initializer
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from utils.tensor_concat import concat_in_fixed_length_two_dimension
from models.lstm_procedure import one_lstm_step, backward_varied_lstm_steps
from models.token_sword_decode import TokenDecoder


class SkeletonOnlyDecodeModel(BasicDecodeModel):
  
  def __init__(self, type_content_data):
    super(SkeletonOnlyDecodeModel, self).__init__(type_content_data)
    
    assert False
    
    number_of_skeletons = self.type_content_data[all_token_summary][SkeletonHitNum]
    self.skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(255, 572, [number_of_skeletons, 2, num_units]))
    self.skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(252, 572, [number_of_skeletons, 2, num_units]))
     
    self.skeleton_lstm_cell = YLSTMCell(1)
    self.skeleton_dup_lstm_cell = YLSTMCell(2)
    self.one_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(258, 578, [number_of_skeletons, num_units]))
    self.skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_hot_skeleton_embedding)
    self.linear_skeleton_output_w = tf.Variable(random_uniform_variable_initializer(257, 576, [number_of_skeletons, num_units]))
    self.one_dup_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(259, 579, [number_of_skeletons, num_units]))
    self.dup_skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_dup_hot_skeleton_embedding)
    
  def set_up_field_when_calling(self, one_example, training):
    self.token_info_tensor = one_example[0]
    self.token_info_start_tensor = one_example[1]
    self.token_info_end_tensor = one_example[2]
    self.token_info_struct_end_tensor = one_example[3]
    self.training = training
    
  def __call__(self, one_example, training = True, ini_metrics = None):
    self.set_up_field_when_calling(one_example, training);
    if ini_metrics == None:
      ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    f_res = tf.while_loop(self.stmt_iterate_cond, self.stmt_iterate_body, [0, tf.shape(self.token_info_start_tensor)[-1], *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
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
    stmt_struct_end = self.token_info_struct_end_tensor[i]
     
    
    stmt_start_offset = 1
    ''' handle skeleton '''
    skt_id = self.token_info_tensor[0][stmt_start]# - skeleton_base
    skt_id_valid_bool = tf.logical_and(tf.greater(skt_id, 2), tf.less(skt_id, self.type_content_data[all_token_summary][SkeletonHitNum]))
    skt_id_valid = tf.cast(skt_id_valid_bool, float_type)
    skt_out_use_id = tf.stack([UNK_en, skt_id])[tf.cast(skt_id_valid_bool, int_type)]
     
    cell = stmt_metrics[self.metrics_index["token_cell"]]
    h = stmt_metrics[self.metrics_index["token_h"]]
    o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(self.training, self.linear_skeleton_output_w, skt_out_use_id, h)
     
    stmt_metrics[self.metrics_index["skeleton_loss"]] = stmt_metrics[self.metrics_index["skeleton_loss"]] + o_loss_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_accurate"]] = stmt_metrics[self.metrics_index["skeleton_accurate"]] + o_accurate_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_mrr"]] = stmt_metrics[self.metrics_index["skeleton_mrr"]] + o_mrr_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_count"]] = stmt_metrics[self.metrics_index["skeleton_count"]] + 1
     
    stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + o_loss_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + o_accurate_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + o_mrr_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + 1
    
    skt_embed = self.skeleton_embedder.compute_h(skt_id)
    _, (next_cell, next_h) = self.skeleton_lstm_cell(skt_embed, (cell, h))
    stmt_metrics[self.metrics_index["token_cell"]] = next_cell
    stmt_metrics[self.metrics_index["token_h"]] = next_h
      
#       if use_dup_model:
#         dup_cell = stmt_metrics[self.metrics_index["dup_token_cell"]]
#         dup_h = stmt_metrics[self.metrics_index["dup_token_h"]]
#         dup_skt_embed = self.dup_skeleton_embedder.compute_h(skt_id)
#         _, (next_dup_cell, next_dup_h) = self.skeleton_dup_lstm_cell(dup_skt_embed, (dup_cell, dup_h))
#         stmt_metrics[self.metrics_index["dup_token_cell"]] = next_dup_cell
#         stmt_metrics[self.metrics_index["dup_token_h"]] = next_dup_h
    
    '''
    this step ignores the statement with no type content tokens (only with skeleton token). 
    '''
    r_stmt_start = stmt_struct_end + 1
    itearate_tokens_continue = tf.cast(stmt_end >= r_stmt_start, int_type)
    f_res = tf.while_loop(self.itearate_tokens_cond, self.itearate_tokens_body, [tf.constant(0, int_type), itearate_tokens_continue, r_stmt_start, stmt_end, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = f_res[4:]
    return (i + 1, i_len, *stmt_metrics)
  
  def skt_iterate_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def skt_iterate_body(self, i, i_len, ini_i, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
#     conserved_memory_length = self.token_info_tensor[3][i]
    if atom_decode_mode == token_decode:
      stmt_metrics = self.token_decoder.decode_one_token(stmt_metrics, self.training, oracle_type_content_en)
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













