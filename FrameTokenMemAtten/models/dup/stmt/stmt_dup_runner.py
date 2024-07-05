import tensorflow as tf
from inputs.example_data_loader import build_statement_dup_feed_dict
from metas.hyper_settings import model_run_mode, statement_dup_mode, top_ks
# from models.dup.skeleton.skeleton_dup_runner import SkeletonDupRunner
from models.dup.stmt.stmt_dup_decoder import StatementDupModel
from models.model_runner import ModelRunner
from metas.non_hyper_constants import int_type, float_type


class StatementDupRunner(ModelRunner):
  
  def __init__(self, sess):
    super(StatementDupRunner, self).__init__(sess)
    
  def set_up_example_loader(self):
    self.example_loader = build_statement_dup_feed_dict
  
  def build_input_place_holder(self):
    self.skeleton_token_info_tensor = tf.compat.v1.placeholder(int_type, [None, None])
    self.skeleton_token_info_start_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.skeleton_token_info_end_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.skeleton_token_base_model_accuracy = tf.compat.v1.placeholder(float_type, [None, len(top_ks)])
    self.skeleton_token_base_model_mrr = tf.compat.v1.placeholder(float_type, [None])
    return [self.skeleton_token_info_tensor, self.skeleton_token_info_start_tensor, self.skeleton_token_info_end_tensor, self.skeleton_token_base_model_accuracy, self.skeleton_token_base_model_mrr]
  
  def build_model_logic(self):
    assert model_run_mode == statement_dup_mode, "serious error! not statement dup mode? but the model logic is statement dup logic."
    self.model = StatementDupModel(self.type_content_data)
  
  def build_feed_dict(self, one_example):
    feed_dict = {self.skeleton_token_info_tensor : one_example[0][0], self.skeleton_token_info_start_tensor : one_example[0][1], self.skeleton_token_info_end_tensor : one_example[0][2], self.skeleton_token_base_model_accuracy : one_example[1], self.skeleton_token_base_model_mrr : one_example[2]}
    return feed_dict
  


