from metas.non_hyper_constants import int_type
import tensorflow as tf
from models.stmt.stmt_decoder import StatementDecodeModel
from models.skeleton.skeleton_only_decoder import SkeletonOnlyDecodeModel


class SkeletonDecodeModel(StatementDecodeModel):
  
  def __init__(self, type_content_data):
    super(SkeletonDecodeModel, self).__init__(type_content_data)
#     number_of_skeletons = self.type_content_data[all_token_summary][SkeletonHitNum]
#     self.skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(255, 572, [number_of_skeletons, 2, num_units]))
#     self.skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(252, 572, [number_of_skeletons, 2, num_units]))
#      
#     self.skeleton_lstm_cell = YLSTMCell(1)
#     self.skeleton_dup_lstm_cell = YLSTMCell(2)
#     self.one_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(258, 578, [number_of_skeletons, num_units]))
#     self.skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_hot_skeleton_embedding)
#     self.linear_skeleton_output_w = tf.Variable(random_uniform_variable_initializer(257, 576, [number_of_skeletons, num_units]))
#     self.one_dup_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(259, 579, [number_of_skeletons, num_units]))
#     self.dup_skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_dup_hot_skeleton_embedding)
    self.skt_only = SkeletonOnlyDecodeModel(type_content_data)
    
    
  def set_up_field_when_calling(self, one_example, training):
    self.token_info_tensor = one_example[0]
    self.token_info_start_tensor = one_example[1]
    self.token_info_end_tensor = one_example[2]
    self.token_info_struct_end_tensor = one_example[3]
    self.training = training
    
  
  def stmt_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    
#     stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
    stmt_struct_end = self.token_info_struct_end_tensor[i]
    
    stmt_metrics = self.skt_only(stmt_metrics, self.token_info_tensor, self.token_info_start_tensor, self.token_info_struct_end_tensor)
#       stmt_start_offset = 1
#       ''' handle skeleton '''
#       skt_id = self.token_info_tensor[0][stmt_start]# - skeleton_base
#       skt_id_valid_bool = tf.logical_and(tf.greater(skt_id, 2), tf.less(skt_id, self.type_content_data[all_token_summary][SkeletonHitNum]))
#       skt_id_valid = tf.cast(skt_id_valid_bool, float_type)
#       skt_out_use_id = tf.stack([UNK_en, skt_id])[tf.cast(skt_id_valid_bool, int_type)]
#        
#       cell = stmt_metrics[self.metrics_index["token_cell"]]
#       h = stmt_metrics[self.metrics_index["token_h"]]
#       o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(self.training, self.linear_skeleton_output_w, skt_out_use_id, h)
#        
#       stmt_metrics[self.metrics_index["skeleton_loss"]] = stmt_metrics[self.metrics_index["skeleton_loss"]] + o_loss_of_this_node * skt_id_valid
#       stmt_metrics[self.metrics_index["skeleton_accurate"]] = stmt_metrics[self.metrics_index["skeleton_accurate"]] + o_accurate_of_this_node * skt_id_valid
#       stmt_metrics[self.metrics_index["skeleton_mrr"]] = stmt_metrics[self.metrics_index["skeleton_mrr"]] + o_mrr_of_this_node * skt_id_valid
#       stmt_metrics[self.metrics_index["skeleton_count"]] = stmt_metrics[self.metrics_index["skeleton_count"]] + 1
#        
#       stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + o_loss_of_this_node * skt_id_valid
#       stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + o_accurate_of_this_node * skt_id_valid
#       stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + o_mrr_of_this_node * skt_id_valid
#       stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + 1
#       
#       skt_embed = self.skeleton_embedder.compute_h(skt_id)
#       _, (next_cell, next_h) = self.skeleton_lstm_cell(skt_embed, (cell, h))
#       stmt_metrics[self.metrics_index["token_cell"]] = next_cell
#       stmt_metrics[self.metrics_index["token_h"]] = next_h
      
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
  
  
  
  







