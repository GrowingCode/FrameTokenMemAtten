from models.model_runner import ModelRunner
import tensorflow as tf
from metas.non_hyper_constants import int_type
from models.tree.tree_decoder import TreeDecodeModel
from metas.hyper_settings import model_run_mode, tree_decode_mode


class TreeModelRunner(ModelRunner):
  
  def __init__(self, sess):
    super(TreeModelRunner, self).__init__(sess)
  
  def build_input_place_holder(self):
    self.post_order_node_type_content_en_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.post_order_node_child_start_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.post_order_node_child_end_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.post_order_node_children_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.pre_post_order_node_type_content_en_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.pre_post_order_node_state_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.pre_post_order_node_post_order_index_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.pre_post_order_node_parent_grammar_index_tensor = tf.compat.v1.placeholder(int_type, [None])
    return (self.post_order_node_type_content_en_tensor, self.post_order_node_child_start_tensor, self.post_order_node_child_end_tensor, self.post_order_node_children_tensor, self.pre_post_order_node_type_content_en_tensor, self.pre_post_order_node_state_tensor, self.pre_post_order_node_post_order_index_tensor, self.pre_post_order_node_parent_grammar_index_tensor)
  
  def build_model_logic(self):
    assert model_run_mode == tree_decode_mode, "serious error! not tree decode mode? but the model logic is tree decode logic."
    self.model = TreeDecodeModel(self.type_content_data)
    
  def build_feed_dict(self, one_example):
    post_order_node_type_content_en = one_example[0]
    post_order_node_child_start = one_example[1]
    post_order_node_child_end = one_example[2]
    post_order_node_children = one_example[3]
    pre_post_order_node_type_content_en = one_example[4]
    pre_post_order_node_state = one_example[5]
    pre_post_order_node_post_order_index = one_example[6]
    pre_post_order_node_parent_grammar_index = one_example[7]
    feed_dict = {self.post_order_node_type_content_en_tensor : post_order_node_type_content_en, 
            self.post_order_node_child_start_tensor : post_order_node_child_start, 
            self.post_order_node_child_end_tensor : post_order_node_child_end, 
            self.post_order_node_children_tensor : post_order_node_children, 
            self.pre_post_order_node_type_content_en_tensor : pre_post_order_node_type_content_en, 
            self.pre_post_order_node_state_tensor : pre_post_order_node_state, 
            self.pre_post_order_node_post_order_index_tensor : pre_post_order_node_post_order_index,
            self.pre_post_order_node_parent_grammar_index_tensor : pre_post_order_node_parent_grammar_index}
    return feed_dict
  
  


