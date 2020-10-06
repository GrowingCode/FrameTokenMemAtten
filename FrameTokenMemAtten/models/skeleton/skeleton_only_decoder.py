from inputs.atom_embeddings import SkeletonAtomEmbed
from metas.hyper_settings import skeleton_decode_way, skeleton_as_one, \
  skeleton_as_pair_encoded, skeleton_as_each, num_units,\
  skeleton_multi_decode_num, skeleton_multi_decode_mode_on, top_ks,\
  skeleton_compute_seq_accurate_on
from metas.non_hyper_constants import float_type, all_token_summary, \
  int_type, SkeletonHitNum, UNK_en, SkeletonPEHitNum, \
  SkeletonEachHitNum, all_skt_one_to_pe_base, all_skt_one_to_pe_start, \
  all_skt_one_to_pe_end, all_skt_one_to_each_base, all_skt_one_to_each_start, \
  all_skt_one_to_each_end
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings,\
  compute_loss_and_accurate_and_top_k_prediction_from_linear_with_computed_embeddings
from models.lstm import YLSTMCell
import tensorflow as tf
from utils.initializer import random_uniform_variable_initializer
from utils.tensor_slice import extract_subsequence_with_start_end_info
from models.token_sequence_decode import compute_beam_sequences, compute_accuracy_of_sequences,\
  dp_compute_en_seqs_from_distinct_parallel_tokens


class SkeletonOnlyDecodeModel():
  
  def __init__(self, type_content_data, metrics_shape, metrics_index):
#     super(SkeletonOnlyDecodeModel, self).__init__(type_content_data)
    self.type_content_data = type_content_data
    self.metrics_shape = metrics_shape
    self.metrics_index = metrics_index
    self.skt_seq_acc_control = True
    
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
      self.skt_info = tf.convert_to_tensor([skt_id], int_type)
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
    
    if self.skt_seq_acc_control and skeleton_compute_seq_accurate_on:
      if skeleton_multi_decode_mode_on:
        stmt_metrics = self.skt_multi_decode(stmt_metrics)
      else:
        stmt_metrics = self.skt_beam_decode(stmt_metrics)
    
    stmt_metrics = tf.while_loop(self.skt_iterate_cond, self.skt_iterate_body, [0, tf.shape(self.skt_info)[-1], *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = list(stmt_metrics[2:])
    
#     f_res = list(post_process_decoder_output(f_res, self.metrics_index))
#     if print_accurate_of_each_example:
#       p_op = tf.print(["accurate:", f_res[self.metrics_index["all_accurate"]], "count:", f_res[self.metrics_index["all_count"]]])
#       with tf.control_dependencies([p_op]):
#         f_res[self.metrics_index["all_count"]] += 0
    return stmt_metrics
  
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
    
#     if not skeleton_multi_decode_mode_on:
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
  
  '''
  the followings are sequence whole decode
  '''
  def skt_beam_decode(self, stmt_metrics):    
    begin_cell = stmt_metrics[self.metrics_index["token_cell"]]
    begin_h = stmt_metrics[self.metrics_index["token_h"]]
    computed_ens = compute_beam_sequences(self.linear_skeleton_output_w, self.skeleton_lstm_cell, self.skeleton_embedder, begin_cell, begin_h, tf.shape(self.skt_info)[-1])
    f_each_acc, f_whole_acc, f_count = compute_accuracy_of_sequences(self.type_content_data, computed_ens, self.skt_info)
    
    stmt_metrics[self.metrics_index["sktseq_as_each_accurate"]] = stmt_metrics[self.metrics_index["sktseq_as_each_accurate"]] + f_each_acc
    stmt_metrics[self.metrics_index["sktseq_as_one_accurate"]] = stmt_metrics[self.metrics_index["sktseq_as_one_accurate"]] + f_whole_acc
    stmt_metrics[self.metrics_index["sktseq_count"]] = stmt_metrics[self.metrics_index["sktseq_count"]] + f_count
    
    return stmt_metrics
  
  def skt_multi_decode(self, stmt_metrics):
#     print("stmt_metrics type:" + str(type(stmt_metrics)))
#     cell = stmt_metrics[self.metrics_index["token_cell"]]
    i = 0
#     p_op = tf.print(['tf.shape(self.skt_info):', tf.shape(self.skt_info)])
#     with tf.control_dependencies([p_op]):
    i_len = tf.minimum(tf.shape(self.skt_info)[-1], skeleton_multi_decode_num)
    
    h = stmt_metrics[self.metrics_index["token_h"]]
    
    o_log_probs = tf.zeros([0, top_ks[-1]], float_type)
    o_ens = tf.zeros([0, top_ks[-1]], int_type)
    
    f_res = tf.while_loop(self.skt_multi_cond, self.skt_multi_body, [i, i_len, h, o_log_probs, o_ens, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([None, top_ks[-1]]), tf.TensorShape([None, top_ks[-1]]), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res)
    
    o_log_probs, o_ens = f_res[3], f_res[4]
    stmt_metrics = f_res[5:]
    
    computed_en_seqs = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
    f_each_acc, f_whole_acc, f_count = compute_accuracy_of_sequences(self.type_content_data, computed_en_seqs, self.skt_info)
    
    stmt_metrics[self.metrics_index["sktseq_as_each_accurate"]] = stmt_metrics[self.metrics_index["sktseq_as_each_accurate"]] + f_each_acc
    stmt_metrics[self.metrics_index["sktseq_as_one_accurate"]] = stmt_metrics[self.metrics_index["sktseq_as_one_accurate"]] + f_whole_acc
    stmt_metrics[self.metrics_index["sktseq_count"]] = stmt_metrics[self.metrics_index["sktseq_count"]] + f_count
    
    return stmt_metrics
  
  def skt_multi_cond(self, i, i_len, *_):
    return tf.less(i, i_len)
  
  def skt_multi_body(self, i, i_len, h, o_log_probs, o_ens, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    transfer_mat = self.skeleton_multi_decode_transfer[i]
    t_h = tf.matmul(h, transfer_mat)
    
    skt_id = self.skt_info[i]
    skt_id_valid_bool = tf.logical_and(tf.greater(skt_id, 2), tf.less(skt_id, self.type_content_data[all_token_summary][SkeletonHitNum]))
    skt_id_valid = tf.cast(skt_id_valid_bool, float_type)
    skt_out_use_id = tf.stack([UNK_en, skt_id])[tf.cast(skt_id_valid_bool, int_type)]
    
    o_log_probs_of_this_node, o_ens_of_this_node, o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_and_top_k_prediction_from_linear_with_computed_embeddings(self.training, self.linear_skeleton_output_w, skt_out_use_id, t_h)
    
    o_log_probs = tf.concat([o_log_probs, [o_log_probs_of_this_node]], axis=0)
    o_ens = tf.concat([o_ens, [o_ens_of_this_node]], axis=0)
    
    stmt_metrics[self.metrics_index["skeleton_loss"]] = stmt_metrics[self.metrics_index["skeleton_loss"]] + o_loss_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_accurate"]] = stmt_metrics[self.metrics_index["skeleton_accurate"]] + o_accurate_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_mrr"]] = stmt_metrics[self.metrics_index["skeleton_mrr"]] + o_mrr_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["skeleton_count"]] = stmt_metrics[self.metrics_index["skeleton_count"]] + 1
    
    stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + o_loss_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + o_accurate_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + o_mrr_of_this_node * skt_id_valid
    stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + 1
    
    return (i+1, i_len, h, o_log_probs, o_ens, *stmt_metrics)
  
  
  
  
  













