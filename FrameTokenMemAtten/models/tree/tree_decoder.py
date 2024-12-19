import tensorflow as tf
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from models.basic_decoder import BasicDecodeModel
from inputs.atom_embeddings import TokenAtomEmbed
from metas.hyper_settings import num_units, top_ks, tree_decode_2d,\
  tree_decode_embed, tree_decode_way, tree_decode_with_grammar,\
  ignore_unk_when_computing_accuracy, tree_decode_without_children,\
  decode_use_cell_mode, decode_use_cell_rnn
from utils.initializer import random_uniform_variable_initializer
from metas.non_hyper_constants import all_token_summary, TokenHitNum, int_type,\
  float_type, UNK_en, all_token_grammar_start, all_token_grammar_end,\
  all_token_grammar_ids
from models.lstm import YLSTMCell, Y2DLSTMCell, YRNNCell
from models.tree.tree_encoder import EncodeOneAST
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings,\
  compute_loss_and_accurate_from_linear_with_computed_embeddings_in_limited_range
from models.token_sword_decode import is_en_valid, is_token_in_consideration


class TreeDecodeModel(BasicDecodeModel):
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    super(TreeDecodeModel, self).__init__(type_content_data)
    if decode_use_cell_mode == decode_use_cell_rnn:
      self.direct_descedent_lstm = YRNNCell(0)
    else:
      self.direct_descedent_lstm = YLSTMCell(0)
    if tree_decode_way == tree_decode_2d:
      self.two_dimen_lstm = Y2DLSTMCell(1)
    else:# tree_decode_way == tree_decode_embed:
      if decode_use_cell_mode == decode_use_cell_rnn:
        self.one_dimen_lstm = YRNNCell(0)
      else:
        self.one_dimen_lstm = YLSTMCell(2)
    
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    self.linear_token_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_tokens, num_units]))
    self.one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_tokens, num_units]))
    self.one_hot_token_cell_embedding = tf.Variable(random_uniform_variable_initializer(25, 56, [number_of_tokens, num_units]))
    self.token_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_embedding)
    self.token_cell_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_cell_embedding)
    
    self.encode_tree = EncodeOneAST(self.type_content_data, self.token_embedder, self.token_cell_embedder)
  
  def create_in_use_tensors_meta(self):
    result = [("token_accumulated_cell", tf.TensorShape([None, num_units])), ("token_accumulated_h", tf.TensorShape([None, num_units]))]
    return result
  
  def __call__(self, one_example, training = True):
    post_order_node_type_content_en_tensor, post_order_node_child_start_tensor, post_order_node_child_end_tensor, post_order_node_children_tensor = one_example[0], one_example[1], one_example[2], one_example[3]
    self.pre_post_order_node_type_content_en_tensor, self.pre_post_order_node_state_tensor, self.pre_post_order_node_post_order_index_tensor, self.pre_post_order_node_parent_grammar_index, self.pre_post_order_node_kind = one_example[4], one_example[5], one_example[6], one_example[7], one_example[8]
    if tree_decode_way != tree_decode_without_children:
      _, self.encoded_h, self.encoded_children_cell, self.encoded_children_h = self.encode_tree.get_encoded_embeds(post_order_node_type_content_en_tensor, post_order_node_child_start_tensor, post_order_node_child_end_tensor, post_order_node_children_tensor)
    self.training = training
    ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    f_res = tf.while_loop(self.tree_iterate_cond, self.tree_iterate_body, [0, tf.shape(self.pre_post_order_node_type_content_en_tensor)[-1], *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2+len(self.statistical_metrics_meta)])
    return f_res
    
  def tree_iterate_cond(self, i, i_len, *_):
      return tf.less(i, i_len)
  
  def tree_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    
    en = self.pre_post_order_node_type_content_en_tensor[i]
    state = self.pre_post_order_node_state_tensor[i]
    post_order_index = self.pre_post_order_node_post_order_index_tensor[i]
    grammar_idx = self.pre_post_order_node_parent_grammar_index[i]
    kind = self.pre_post_order_node_kind[i]
    
#     if token_accuracy_mode == consider_all_token_accuracy:
#       t_valid_bool = tf.constant(True, bool_type)
#     elif token_accuracy_mode == only_consider_token_kind_accuracy:
#       t_valid_bool = is_in_token_kind_range(kind)
#     else:
#       assert False
#     t_valid_float = tf.cast(t_valid_bool, float_type)
#     t_valid_int = tf.cast(t_valid_bool, int_type)
    t_valid_float, t_valid_int = is_token_in_consideration(en, -1, kind, self.type_content_data[all_token_summary][TokenHitNum])
    en_valid_float, en_valid_int = is_en_valid(en, self.type_content_data[all_token_summary][TokenHitNum])
#     en_valid_bool = tf.logical_and(tf.greater(en, 2), tf.less(en, self.type_content_data[all_token_summary][TokenHitNum]))
#     en_valid_float = tf.cast(en_valid_bool, float_type)
#     en_valid_int = tf.cast(en_valid_bool, int_type)
    out_use_en = tf.stack([UNK_en, en])[en_valid_int]
    
    non_leaf_post_bool = tf.equal(state, 2)
    non_leaf_post = tf.cast(non_leaf_post_bool, int_type)
    node_acc_valid = 1 - non_leaf_post
    
    ''' pre remove '''
    before_remove_length = tf.shape(stmt_metrics[self.metrics_index["token_accumulated_cell"]])[0]
    stmt_metrics[self.metrics_index["token_accumulated_cell"]] = tf.slice(stmt_metrics[self.metrics_index["token_accumulated_cell"]], [0, 0], [before_remove_length - non_leaf_post, num_units])
    stmt_metrics[self.metrics_index["token_accumulated_h"]] = tf.slice(stmt_metrics[self.metrics_index["token_accumulated_h"]], [0, 0], [before_remove_length - non_leaf_post, num_units])
    
    cell = tf.convert_to_tensor([stmt_metrics[self.metrics_index["token_accumulated_cell"]][-1]])
    h = tf.convert_to_tensor([stmt_metrics[self.metrics_index["token_accumulated_h"]][-1]])
    p_a_h = h
    
    if (not self.training) and tree_decode_with_grammar:
      start_idx = self.type_content_data[all_token_grammar_start][grammar_idx]
      end_idx = self.type_content_data[all_token_grammar_end][grammar_idx]
      ens_range = tf.slice(self.type_content_data[all_token_grammar_ids], [start_idx], [end_idx-start_idx+1])
      o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings_in_limited_range(self.training, self.linear_token_output_w, ens_range, out_use_en, p_a_h)
    else:
      o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(self.training, self.linear_token_output_w, out_use_en, p_a_h)
      
    mrr_of_this_node = tf.stack([0.0, o_mrr_of_this_node])[node_acc_valid]
    accurate_of_this_node = tf.stack([tf.zeros([len(top_ks)], float_type), o_accurate_of_this_node])[node_acc_valid]
    loss_of_this_node = tf.stack([0.0, o_loss_of_this_node])[node_acc_valid]
    
    count_of_this_node = tf.stack([0, 1])[node_acc_valid]
    r_count = count_of_this_node * t_valid_int
    if ignore_unk_when_computing_accuracy:
      r_count = r_count * en_valid_int
    
    stmt_metrics[self.metrics_index["token_loss"]] = stmt_metrics[self.metrics_index["token_loss"]] + loss_of_this_node * en_valid_float
    stmt_metrics[self.metrics_index["token_accurate"]] = stmt_metrics[self.metrics_index["token_accurate"]] + accurate_of_this_node * en_valid_float * t_valid_float
    stmt_metrics[self.metrics_index["token_mrr"]] = stmt_metrics[self.metrics_index["token_mrr"]] + mrr_of_this_node * en_valid_float * t_valid_float
    stmt_metrics[self.metrics_index["token_count"]] = stmt_metrics[self.metrics_index["token_count"]] + r_count
    stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + loss_of_this_node * en_valid_float
    stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + accurate_of_this_node * en_valid_float * t_valid_float
    stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + mrr_of_this_node * en_valid_float * t_valid_float
    stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + r_count
    
    stmt_metrics[self.metrics_index["token_accurate_each_noavg"]] = stmt_metrics[self.metrics_index["token_accurate_each_noavg"]].write(stmt_metrics[self.metrics_index["token_accurate_each_noavg"]].size(), accurate_of_this_node * en_valid_float * t_valid_float)
    stmt_metrics[self.metrics_index["token_mrr_each_noavg"]] = stmt_metrics[self.metrics_index["token_mrr_each_noavg"]].write(stmt_metrics[self.metrics_index["token_mrr_each_noavg"]].size(), mrr_of_this_node * en_valid_float * t_valid_float)
    
    stmt_metrics[self.metrics_index["token_count_each_int_noavg"]] = stmt_metrics[self.metrics_index["token_count_each_int_noavg"]].write(stmt_metrics[self.metrics_index["token_count_each_int_noavg"]].size(), r_count)
    
    ''' infer next cell/h '''
#       p_op = tf.print("loss_of_this_node:", loss_of_this_node, "accurate_of_this_node:", accurate_of_this_node)
#       with tf.control_dependencies([p_op]):
    en_h = self.token_embedder.compute_h(en)
    _, (next_cell1, next_h1) = self.direct_descedent_lstm(en_h, (cell, h))
    if tree_decode_way == tree_decode_2d:
      next_cell2, next_h2 = self.two_dimen_lstm(en_h, cell, h, [self.encoded_children_cell[post_order_index]], [self.encoded_children_h[post_order_index]])
    elif tree_decode_way == tree_decode_embed:
      _, (next_cell2, next_h2) = self.one_dimen_lstm(tf.expand_dims(self.encoded_h[post_order_index], axis=0), (cell, h))
    elif tree_decode_way == tree_decode_without_children:
      _, (next_cell2, next_h2) = self.one_dimen_lstm(en_h, (cell, h))
    else:
      print("Unrecognized tree decode mode!")
      assert False
    next_cell = tf.stack([next_cell1, next_cell2])[non_leaf_post]
    next_h = tf.stack([next_h1, next_h2])[non_leaf_post]
    
    ''' update accumulated cell/h '''
    ''' post remove '''
    should_remove = tf.cast(tf.greater_equal(state, 1), int_type)
    after_remove_length = tf.shape(stmt_metrics[self.metrics_index["token_accumulated_cell"]])[0]
    stmt_metrics[self.metrics_index["token_accumulated_cell"]] = tf.slice(stmt_metrics[self.metrics_index["token_accumulated_cell"]], [0, 0], [after_remove_length - should_remove, num_units])
    stmt_metrics[self.metrics_index["token_accumulated_h"]] = tf.slice(stmt_metrics[self.metrics_index["token_accumulated_h"]], [0, 0], [after_remove_length - should_remove, num_units])
    ''' concatenate newly inferred '''
    stmt_metrics[self.metrics_index["token_accumulated_cell"]] = tf.concat([stmt_metrics[self.metrics_index["token_accumulated_cell"]], next_cell], axis=0)
    stmt_metrics[self.metrics_index["token_accumulated_h"]] = tf.concat([stmt_metrics[self.metrics_index["token_accumulated_h"]], next_h], axis=0)
    
    return (i+1, i_len, *stmt_metrics)
  
  





