from metas.non_hyper_constants import int_type
import tensorflow as tf
from models.dup.stmt.stmt_dup_decoder import StatementDupModel
from models.skeleton.skeleton_only_decoder import SkeletonOnlyDecodeModel


class SkeletonDupModel(StatementDupModel):
  
  def __init__(self, type_content_data):
    super(SkeletonDupModel, self).__init__(type_content_data)
    
    self.skt_only = SkeletonOnlyDecodeModel(type_content_data, self.metrics_shape, self.metrics_index)
    
#     self.treat_first_element_as_skeleton = 1
    
#     number_of_skeletons = self.type_content_data[all_token_summary][SkeletonHitNum]
#      
#     if self.treat_first_element_as_skeleton:
#       self.skeleton_dup_lstm_cell = YLSTMCell(2)
#       self.one_dup_hot_skeleton_embedding = tf.Variable(random_uniform_variable_initializer(259, 579, [number_of_skeletons, num_units]))
#       self.dup_skeleton_embedder = SkeletonAtomEmbed(self.type_content_data, self.one_dup_hot_skeleton_embedding)
     
#     number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
#     
#     if atom_decode_mode == token_decode:
#       self.one_dup_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(252, 226, [number_of_tokens, num_units]))
#       self.dup_token_embedder = TokenAtomEmbed(self.type_content_data, self.one_dup_hot_token_embedding)
#       self.dup_token_lstm = YLSTMCell(9)
#       self.dup_token_pointer = PointerNetwork(655)
#       self.dup_skeleton_forward_cell_h = tf.Variable(random_uniform_variable_initializer(155, 572, [number_of_skeletons, 2, num_units]))
#       self.dup_skeleton_backward_cell_h = tf.Variable(random_uniform_variable_initializer(152, 572, [number_of_skeletons, 2, num_units]))
#       self.integrate_computer = None
#       if compute_memory_in_only_memory_mode:
#         self.integrate_computer = Y2DirectLSTMCell(105)
#       if compute_token_memory:
#         self.dup_mem_nn = NTMOneDirection(800)
#         self.dup_forward_token_lstm = YLSTMCell(10)
#         self.dup_backward_token_lstm = YLSTMCell(11)
#       self.dup_token_decoder = DupTokenDecoder(type_content_data, self.metrics_index, self.dup_token_pointer) 
#     else:
#       assert False, "Wrong atom_decode_mode"
  
#   def __call__(self, one_example, training=True):
#     self.token_info_tensor = one_example[0]
#     self.token_info_start_tensor = one_example[1]
#     self.token_info_end_tensor = one_example[2]
#     self.token_base_model_accuracy = one_example[4]
#     self.token_base_model_mrr = one_example[5]
#     self.training = training
#     ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
#     f_res = tf.while_loop(self.stmt_iterate_cond, self.stmt_iterate_body, [0, tf.shape(self.token_info_start_tensor)[-1], *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
#     f_res = list(f_res[2:2 + len(self.statistical_metrics_meta)])
#     if print_accurate_of_each_example:
#       p_op = tf.print(["accurate:", f_res[self.metrics_index["all_accurate"]], "count:", f_res[self.metrics_index["all_count"]]])
#       with tf.control_dependencies([p_op]):
#         f_res[self.metrics_index["all_count"]] += 0
#     return f_res
  
  def create_in_use_tensors_meta(self):
    result = super(SkeletonDupModel, self).create_in_use_tensors_meta() + [("skt_e_int_noavg", tf.TensorShape(None)), ("skt_pe_int_noavg", tf.TensorShape(None))]
    return result
  
  def stmt_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
     
    stmt_start = self.token_info_start_tensor[i]
    stmt_end = self.token_info_end_tensor[i]
#     stmt_struct_end = self.token_info_struct_end_tensor[i]
    
    stmt_metrics = self.skt_only(stmt_metrics, self.token_info_tensor, stmt_start)#, self.token_base_model_accuracy, self.token_base_model_mrr
    
#     stmt_start_offset = 0
#     if self.treat_first_element_as_skeleton:
#       stmt_start_offset = 1
#       ''' handle skeleton '''
#       skt_id = self.token_info_tensor[0][stmt_start]# - skeleton_base
#        
#       dup_cell = stmt_metrics[self.metrics_index["dup_token_cell"]]
#       dup_h = stmt_metrics[self.metrics_index["dup_token_h"]]
#       dup_skt_embed = self.dup_skeleton_embedder.compute_h(skt_id)
#       _, (next_dup_cell, next_dup_h) = self.skeleton_dup_lstm_cell(dup_skt_embed, (dup_cell, dup_h))
#       stmt_metrics[self.metrics_index["dup_token_cell"]] = next_dup_cell
#       stmt_metrics[self.metrics_index["dup_token_h"]] = next_dup_h
#     else:
#       stmt_start_offset = 0
    
    '''
    this step ignores the statement with no type content tokens (only with skeleton token). 
    '''
    r_stmt_start = stmt_start + 1
    itearate_tokens_continue = tf.cast(stmt_end >= r_stmt_start, int_type)
    f_res = tf.while_loop(self.itearate_tokens_cond, self.itearate_tokens_body, [tf.constant(0, int_type), itearate_tokens_continue, r_stmt_start, stmt_end, *stmt_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    stmt_metrics = f_res[4:]
    return (i + 1, i_len, *stmt_metrics)
  
#   def token_iterate_cond(self, i, i_len, *_):
#     return tf.less_equal(i, i_len)
#   
#   def forward_loop_cond(self, i, i_len, *_):
#     return tf.less_equal(i, i_len)
#   
#   def forward_loop_body(self, i, i_len, dup_embeds, *stmt_metrics_tuple):
#     stmt_metrics = list(stmt_metrics_tuple)
#     dup_f_cell = [stmt_metrics[self.metrics_index["dup_loop_forward_cells"]][-1]]
#     dup_f_h = [stmt_metrics[self.metrics_index["dup_loop_forward_hs"]][-1]]
#     _, (new_dup_f_cell, new_dup_f_h) = self.dup_forward_token_lstm(tf.expand_dims(dup_embeds[i], axis=0), (dup_f_cell, dup_f_h))
#     stmt_metrics[self.metrics_index["dup_loop_forward_cells"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_cells"]], new_dup_f_cell, -1)
#     stmt_metrics[self.metrics_index["dup_loop_forward_hs"]] = concat_in_fixed_length_two_dimension(stmt_metrics[self.metrics_index["dup_loop_forward_hs"]], new_dup_f_h, -1)
#     return (i + 1, i_len, dup_embeds, *stmt_metrics)
#   
#   def backward_loop_cond(self, i, i_len, *_):
#     return tf.less_equal(i, i_len)
#   
#   def backward_loop_body(self, i, i_len, dup_embeds, *stmt_metrics_tuple):
#     stmt_metrics = list(stmt_metrics_tuple)
#     dup_b_cell = [stmt_metrics[self.metrics_index["dup_loop_backward_cells"]][0]]
#     dup_b_h = [stmt_metrics[self.metrics_index["dup_loop_backward_hs"]][0]]
#     _, (new_dup_b_cell, new_dup_b_h) = self.dup_backward_token_lstm(tf.expand_dims(dup_embeds[i_len], axis=0), (dup_b_cell, dup_b_h))
#     stmt_metrics[self.metrics_index["dup_loop_backward_cells"]] = concat_in_fixed_length_two_dimension(new_dup_b_cell, stmt_metrics[self.metrics_index["dup_loop_backward_cells"]], -1)
#     stmt_metrics[self.metrics_index["dup_loop_backward_hs"]] = concat_in_fixed_length_two_dimension(new_dup_b_h, stmt_metrics[self.metrics_index["dup_loop_backward_hs"]], -1)
#     return (i, i_len - 1, dup_embeds, *stmt_metrics)






