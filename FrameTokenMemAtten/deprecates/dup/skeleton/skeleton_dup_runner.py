from models.model_runner import ModelRunner
import tensorflow as tf
from metas.non_hyper_constants import int_type, float_type
from metas.hyper_settings import model_run_mode, top_ks,\
  skeleton_dup_mode
from models.dup.skeleton.skeleton_dup_decoder import SkeletonDupModel
from inputs.example_data_loader import build_skeleton_dup_feed_dict


class SkeletonDupRunner(ModelRunner):
  
  def __init__(self, sess):
    super(SkeletonDupRunner, self).__init__(sess)
  
  def set_up_example_loader(self):
    self.example_loader = build_skeleton_dup_feed_dict
  
  def build_input_place_holder(self):
    self.skeleton_token_info_tensor = tf.compat.v1.placeholder(int_type, [None, None])
    self.skeleton_token_info_start_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.skeleton_token_info_end_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.skeleton_token_base_model_accuracy = tf.compat.v1.placeholder(float_type, [None, len(top_ks)])
    self.skeleton_token_base_model_mrr = tf.compat.v1.placeholder(float_type, [None])
    return [self.skeleton_token_info_tensor, self.skeleton_token_info_start_tensor, self.skeleton_token_info_end_tensor, self.skeleton_token_base_model_accuracy, self.skeleton_token_base_model_mrr]
  
  def build_model_logic(self):
    assert model_run_mode == skeleton_dup_mode, "serious error! not skeleton dup mode? but the model logic is skeleton dup logic."
    self.model = SkeletonDupModel(self.type_content_data)
    
  def build_feed_dict(self, one_example):
    feed_dict = {self.skeleton_token_info_tensor : one_example[0][0], self.skeleton_token_info_start_tensor : one_example[0][1], self.skeleton_token_info_end_tensor : one_example[0][2], self.skeleton_token_base_model_accuracy : one_example[1], self.skeleton_token_base_model_mrr : one_example[2]}
    return feed_dict
  
  


