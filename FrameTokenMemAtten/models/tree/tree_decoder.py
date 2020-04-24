import tensorflow as tf
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from models.basic_decoder import BasicDecodeModel
from inputs.atom_embeddings import TokenAtomEmbed
from metas.hyper_settings import num_units, top_ks, tree_decode_2d,\
  tree_decode_embed, tree_decode_way
from utils.initializer import random_uniform_variable_initializer
from metas.non_hyper_constants import all_token_summary, TokenHitNum, int_type,\
  float_type, UNK_en, all_token_grammar_start, all_token_grammar_end,\
  all_token_grammar_ids
from models.lstm import YLSTMCell, Y2DLSTMCell
from models.tree.tree_encoder import EncodeOneAST
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings,\
  compute_loss_and_accurate_from_linear_with_computed_embeddings_in_limited_range


class TreeDecodeModel(BasicDecodeModel):
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    super(TreeDecodeModel, self).__init__(type_content_data)
    self.direct_descedent_lstm = YLSTMCell(0)
    if tree_decode_way == tree_decode_2d:
      self.two_dimen_lstm = Y2DLSTMCell(1)
    elif tree_decode_way == tree_decode_embed:
      self.one_dimen_lstm = YLSTMCell(2)
    
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    self.linear_token_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_tokens, num_units]))
    self.one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_tokens, num_units]))
    self.one_hot_token_cell_embedding = tf.Variable(random_uniform_variable_initializer(25, 56, [number_of_tokens, num_units]))
    self.token_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_embedding)
    self.token_cell_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_cell_embedding)
    
    self.encode_tree = EncodeOneAST(self.type_content_data, self.token_embedder, self.token_cell_embedder)
  
  def __call__(self, one_example, training = True):
    post_order_node_type_content_en_tensor, post_order_node_child_start_tensor, post_order_node_child_end_tensor, post_order_node_children_tensor = one_example[0], one_example[1], one_example[2], one_example[3]
    self.pre_post_order_node_type_content_en_tensor, self.pre_post_order_node_state_tensor, self.pre_post_order_node_post_order_index_tensor, self.pre_post_order_node_parent_grammar_index = one_example[4], one_example[5], one_example[6], one_example[7]
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
    
    en_valid_bool = tf.logical_and(tf.greater(en, 2), tf.less(en, self.type_content_data[all_token_summary][TokenHitNum]))
    out_use_en = tf.stack([UNK_en, en])[tf.cast(en_valid_bool, int_type)]
    
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
    
    if self.training:
      r_use_output_w = self.linear_token_output_w
      o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(self.training, r_use_output_w, out_use_en, p_a_h)
    else:
      start_idx = self.type_content_data[all_token_grammar_start][grammar_idx]
      end_idx = self.type_content_data[all_token_grammar_end][grammar_idx]
      out_range = tf.slice(self.type_content_data[all_token_grammar_ids], [start_idx], [end_idx-start_idx+1])
      r_use_output_w = tf.gather(self.linear_token_output_w, out_range)
      o_mrr_of_this_node, o_accurate_of_this_node, o_loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings_in_limited_range(self.training, out_range, r_use_output_w, out_use_en, p_a_h)
    
    mrr_of_this_node = tf.stack([0.0, o_mrr_of_this_node])[node_acc_valid]
    accurate_of_this_node = tf.stack([tf.zeros([len(top_ks)], float_type), o_accurate_of_this_node])[node_acc_valid]
    loss_of_this_node = tf.stack([0.0, o_loss_of_this_node])[node_acc_valid]
    count_of_this_node = tf.stack([0, 1])[node_acc_valid]
    stmt_metrics[self.metrics_index["token_loss"]] = stmt_metrics[self.metrics_index["token_loss"]] + loss_of_this_node
    stmt_metrics[self.metrics_index["token_accurate"]] = stmt_metrics[self.metrics_index["token_accurate"]] + accurate_of_this_node
    stmt_metrics[self.metrics_index["token_mrr"]] = stmt_metrics[self.metrics_index["token_mrr"]] + mrr_of_this_node
    stmt_metrics[self.metrics_index["token_count"]] = stmt_metrics[self.metrics_index["token_count"]] + count_of_this_node
    stmt_metrics[self.metrics_index["all_loss"]] = stmt_metrics[self.metrics_index["all_loss"]] + loss_of_this_node
    stmt_metrics[self.metrics_index["all_accurate"]] = stmt_metrics[self.metrics_index["all_accurate"]] + accurate_of_this_node
    stmt_metrics[self.metrics_index["all_mrr"]] = stmt_metrics[self.metrics_index["all_mrr"]] + mrr_of_this_node
    stmt_metrics[self.metrics_index["all_count"]] = stmt_metrics[self.metrics_index["all_count"]] + count_of_this_node
    
    ''' infer next cell/h '''
#       p_op = tf.print("loss_of_this_node:", loss_of_this_node, "accurate_of_this_node:", accurate_of_this_node)
#       with tf.control_dependencies([p_op]):
    en_h = self.token_embedder.compute_h(en)
    next_cell1, next_h1 = self.direct_descedent_lstm(en_h, cell, h)
    if tree_decode_way == tree_decode_2d:
      next_cell2, next_h2 = self.two_dimen_lstm(en_h, cell, h, [self.encoded_children_cell[post_order_index]], [self.encoded_children_h[post_order_index]])
    elif tree_decode_way == tree_decode_embed:
      next_cell2, next_h2 = self.one_dimen_lstm([self.encoded_h[post_order_index]], cell, h)
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
  
  





