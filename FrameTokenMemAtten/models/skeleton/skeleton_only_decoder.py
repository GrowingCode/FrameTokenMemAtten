from inputs.atom_embeddings import SkeletonAtomEmbed
from metas.hyper_settings import skeleton_decode_way, skeleton_as_one, \
  skeleton_as_pair_encoded, skeleton_as_each, num_units,\
  skeleton_multi_decode_num, skeleton_multi_decode_mode_on
from metas.non_hyper_constants import float_type, all_token_summary, \
  int_type, SkeletonHitNum, UNK_en, SkeletonPEHitNum, \
  SkeletonEachHitNum, all_skt_one_to_pe_base, all_skt_one_to_pe_start, \
  all_skt_one_to_pe_end, all_skt_one_to_each_base, all_skt_one_to_each_start, \
  all_skt_one_to_each_end
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings
from models.lstm import YLSTMCell
from models.stmt.stmt_decoder import StatementDecodeModel
import tensorflow as tf
from utils.initializer import random_uniform_variable_initializer
from utils.tensor_slice import extract_subsequence_with_start_end_info


class SkeletonOnlyDecodeModel(StatementDecodeModel):
  
  def __init__(self, type_content_data, compute_noavg = True):
    super(SkeletonOnlyDecodeModel, self).__init__(type_content_data)
    self.compute_noavg = compute_noavg
    
    assert False
    
    number_of_skeletons = self.type_content_data[all_token_summary][SkeletonHitNum]
    pe_number_of_skeletons = self.type_content_data[all_token_summary][SkeletonPEHitNum]
    each_number_of_skeletons = self.type_content_data[all_token_summary][SkeletonEachHitNum]
    
    if skeleton_decode_way == skeleton_as_one:
      vocab_num = number_of_skeletons
    elif skeleton_decode_way == skeleton_as_pair_encoded:
      vocab_num = pe_number_of_skeletons
    elif skeleton_decode_way == skeleton_as_each:
      vocab_num = each_number_of_skeletons
    else:
      assert False, "Unrecognized skeleton_decode_way!"
    self.vocab_num = vocab_num
    
    self.initialize_parameters();
    
  def initialize_parameters(self):
    
    self.skeleton_lstm_cell = YLSTMCell(1)
    
    self.one_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(258, 578, [self.vocab_num, num_units]))
    self.skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_hot_skeleton_embedding, self.vocab_num)
    self.linear_skeleton_output_w = tf.Variable(random_uniform_variable_initializer(257, 576, [self.vocab_num, num_units]))
    
    if skeleton_multi_decode_mode_on:
      self.skeleton_multi_decode_transfer = tf.Variable(random_uniform_variable_initializer(527, 567, [skeleton_multi_decode_num, num_units, num_units]))
    
  def __call__(self, stmt_metrics, skt_id, training = True):
    
    self.training = training
    
    if skeleton_decode_way == skeleton_as_one:
      self.skt_info = tf.convert_to_tensor([skt_id])
    elif skeleton_decode_way == skeleton_as_pair_encoded:
      ope_base = self.type_content_data[all_skt_one_to_pe_base]
      ope_start = self.type_content_data[all_skt_one_to_pe_start]
      ope_end = self.type_content_data[all_skt_one_to_pe_end]
      self.skt_info = extract_subsequence_with_start_end_info(ope_base, ope_start[skt_id], ope_end[skt_id])
    elif skeleton_decode_way == skeleton_as_each:
      oe_base = self.type_content_data[all_skt_one_to_each_base]
      oe_start = self.type_content_data[all_skt_one_to_each_start]
      oe_end = self.type_content_data[all_skt_one_to_each_end]
      self.skt_info = extract_subsequence_with_start_end_info(oe_base, oe_start[skt_id], oe_end[skt_id])
    
    f_res = tf.while_loop(self.skt_iterate_cond, self.skt_iterate_body, [0, tf.shape(self.skt_info)[-1], *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2+len(self.statistical_metrics_meta)])
#     f_res = list(post_process_decoder_output(f_res, self.metrics_index))
#     if print_accurate_of_each_example:
#       p_op = tf.print(["accurate:", f_res[self.metrics_index["all_accurate"]], "count:", f_res[self.metrics_index["all_count"]]])
#       with tf.control_dependencies([p_op]):
#         f_res[self.metrics_index["all_count"]] += 0
    return f_res
  
  def skt_iterate_cond(self, i, i_len, *_):
    return tf.less(i, i_len)
  
  def skt_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    
#     stmt_struct_end = self.token_info_struct_end_tensor[i]
#     stmt_start_offset = 1
    ''' handle skeleton '''
    skt_id = self.skt_info[i]# - skeleton_base
    
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
    
    return (i + 1, i_len, *stmt_metrics)
  
  def skt_multi_decode(self, *stmt_metrics_tuple):
    
    stmt_metrics = list(stmt_metrics_tuple)
    cell = stmt_metrics[self.metrics_index["token_cell"]]
    h = stmt_metrics[self.metrics_index["token_h"]]
    
    i = 0
    i_len = tf.minimum(tf.shape(self.skt_info)[-1], skeleton_multi_decode_num)
    
    _, _, _, stmt_metrics = tf.while_loop(self.skt_multi_cond, self.skt_multi_body, [i, i_len, h, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), *self.metrics_shape], parallel_iterations=1)
    
    
    pass
  
  def skt_multi_cond(self, i, i_len, *_):
    return tf.less(i, i_len)
  
  def skt_multi_body(self, i, i_len, h, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    transfer_mat = self.skeleton_multi_decode_transfer[i]
    t_h = tf.matmul(h, transfer_mat)
    
    skt_id = self.skt_info[i]
    skt_id_valid_bool = tf.logical_and(tf.greater(skt_id, 2), tf.less(skt_id, self.type_content_data[all_token_summary][SkeletonHitNum]))
    skt_id_valid = tf.cast(skt_id_valid_bool, float_type)
    skt_out_use_id = tf.stack([UNK_en, skt_id])[tf.cast(skt_id_valid_bool, int_type)]
    
    o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(self.training, self.linear_skeleton_output_w, skt_out_use_id, h)
    
    stmt_metrics[self.metrics_index["skeleton_loss"]] = stmt_metrics[self.metrics_index["skeleton_loss"]] + o_loss_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_accurate"]] = stmt_metrics[self.metrics_index["skeleton_accurate"]] + o_accurate_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_mrr"]] = stmt_metrics[self.metrics_index["skeleton_mrr"]] + o_mrr_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_count"]] = stmt_metrics[self.metrics_index["skeleton_count"]] + 1
    
    stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + o_loss_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + o_accurate_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + o_mrr_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + 1
    
    
    return i+1, i_len, stmt_metrics
  
  
  
  
  













