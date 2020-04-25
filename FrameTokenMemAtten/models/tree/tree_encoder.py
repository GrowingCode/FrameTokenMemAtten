import tensorflow as tf
from models.lstm import Y2DirectLSTMCell, Y2DLSTMCell, YLSTMCell
from metas.non_hyper_constants import int_type
from metas.hyper_settings import num_units, tree_leaf_one_more_lstm_step
from metas.tensor_constants import zero_tensor
from utils.initializer import random_uniform_variable_initializer


class EncodeOneAST():
  
  def __init__(self, type_content_data, token_embedder, token_cell_embedder):
    self.type_content_data = type_content_data
    self.token_embedder = token_embedder
    self.token_cell_embedder = token_cell_embedder
    self.y2direct_lstm = Y2DirectLSTMCell(60)
    self.y2d_lstm = Y2DLSTMCell(61)
    self.y_lstm = YLSTMCell(62)
    if tree_leaf_one_more_lstm_step:
      self.y_leaf_lstm = YLSTMCell(63)
    self.y_forward = YLSTMCell(64)
    self.y_backward = YLSTMCell(65)
    self.tree_leaf_one_more_lstm_begin_cell_h = tf.Variable(random_uniform_variable_initializer(25, 5, [2, num_units]))
    
  def get_encoded_embeds(self, post_order_node_type_content_en, post_order_node_child_start, post_order_node_child_end, post_order_node_children):
    
    def encode_cond(i, i_len, *_):
      return tf.less(i, i_len)
    
    def encode_body(i, i_len, encoded_cell, encoded_h, encoded_children_cell, encoded_children_h):
      
      def iterate_children_forward_cond(j, j_len, *_):
        return tf.logical_and(tf.greater_equal(j_len, tf.constant(0, int_type)), tf.less_equal(j, j_len))
      
      def iterate_children_forward_body(j, j_len, f_cell, f_h):
        _, (f_cell, f_h) = self.y_forward(tf.expand_dims(encoded_h[post_order_node_children[j]], axis=0), (f_cell, f_h))
        return j+1, j_len, f_cell, f_h
      
      def iterate_children_backward_cond(j, j_len, *_):
        return tf.logical_and(tf.greater_equal(j_len, tf.constant(0, int_type)), tf.less_equal(j, j_len))
      
      def iterate_children_backward_body(j, j_len, b_cell, b_h):
        _, (b_cell, b_h) = self.y_backward(tf.expand_dims(encoded_h[post_order_node_children[j_len]], axis=0), (b_cell, b_h))
        return j, j_len-1, b_cell, b_h
      
      '''
      encode children
      '''
      child_start = post_order_node_child_start[i]
      child_end = post_order_node_child_end[i]
      c_select = tf.cast(tf.greater_equal(child_end, child_start), int_type)
      c2_select = tf.cast(tf.greater(child_end, child_start), int_type)
      child_start_index = tf.stack([0, child_start])[c_select]
      child_end_index = tf.stack([0, child_end])[c_select]
      f_cell = tf.stack([zero_tensor, [encoded_cell[post_order_node_children[child_start_index]]]])[c_select]
      f_h = tf.stack([zero_tensor, [encoded_h[post_order_node_children[child_start_index]]]])[c_select]
      b_cell = tf.stack([zero_tensor, [encoded_cell[post_order_node_children[child_end_index]]]])[c_select]
      b_h = tf.stack([zero_tensor, [encoded_h[post_order_node_children[child_end_index]]]])[c_select]
      _, _, f_cell, f_h = tf.while_loop(iterate_children_forward_cond, iterate_children_forward_body, [child_start+1, child_end, f_cell, f_h], parallel_iterations=1)
      _, _, b_cell, b_h = tf.while_loop(iterate_children_backward_cond, iterate_children_backward_body, [child_start, child_end-1, b_cell, b_h], parallel_iterations=1)
      '''
      encode self type/content
      '''
      en = post_order_node_type_content_en[i]
      en_cell = self.token_cell_embedder.compute_h(en)
      en_h = self.token_embedder.compute_h(en)
      
      if tree_leaf_one_more_lstm_step:
        _, (en_cell, en_h) = self.y_leaf_lstm(en_h, (tf.expand_dims(self.tree_leaf_one_more_lstm_begin_cell_h[0], axis=0), tf.expand_dims(self.tree_leaf_one_more_lstm_begin_cell_h[1], axis=0)))
      
      _, (r_x_cell, r_x_h) = self.y_lstm(en_h, (f_cell, f_h))
      r_x_cell2, r_x_h2 = self.y2d_lstm(en_h, f_cell, f_h, b_cell, b_h)
      x_cell = tf.stack([en_cell, tf.stack([r_x_cell, r_x_cell2])[c2_select]])[c_select]
      x_h = tf.stack([en_h, tf.stack([r_x_h, r_x_h2])[c2_select]])[c_select]
      encoded_cell = tf.concat([encoded_cell, x_cell], axis=0)
      encoded_h = tf.concat([encoded_h, x_h], axis=0)
      x_c_cell, x_c_h = f_cell, f_h
      x_c_cell2, x_c_h2 = self.y2direct_lstm(f_cell, f_h, b_cell, b_h)
      c_cell = tf.stack([zero_tensor, tf.stack([x_c_cell, x_c_cell2])[c2_select]])[c_select]
      c_h = tf.stack([zero_tensor, tf.stack([x_c_h, x_c_h2])[c2_select]])[c_select]
      encoded_children_cell = tf.concat([encoded_children_cell, c_cell], axis=0)
      encoded_children_h = tf.concat([encoded_children_h, c_h], axis=0)
      return i+1, i_len, encoded_cell, encoded_h, encoded_children_cell, encoded_children_h
    
    encoded_cell, encoded_h, encoded_children_cell, encoded_children_h = zero_tensor, zero_tensor, zero_tensor, zero_tensor
    _, _, encoded_cell, encoded_h, encoded_children_cell, encoded_children_h = tf.while_loop(encode_cond, encode_body, [1, tf.shape(post_order_node_child_start)[-1], encoded_cell, encoded_h, encoded_children_cell, encoded_children_h], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units])], parallel_iterations=1)
    return encoded_cell, encoded_h, encoded_children_cell, encoded_children_h
  
