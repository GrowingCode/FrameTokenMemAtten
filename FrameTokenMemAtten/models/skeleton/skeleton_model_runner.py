from models.model_runner import ModelRunner
import tensorflow as tf
from metas.non_hyper_constants import int_type
from metas.hyper_settings import model_run_mode,\
  skeleton_decode_mode
from models.skeleton.skeleton_decoder import SkeletonDecodeModel


class SkeletonModelRunner(ModelRunner):
  
  def __init__(self, sess):
    super(SkeletonModelRunner, self).__init__(sess)
  
  def build_input_place_holder(self):
    self.skeleton_token_info_tensor = tf.compat.v1.placeholder(int_type, [None, None])
    self.skeleton_token_info_start_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.skeleton_token_info_end_tensor = tf.compat.v1.placeholder(int_type, [None])
    return (self.skeleton_token_info_tensor, self.skeleton_token_info_start_tensor, self.skeleton_token_info_end_tensor)
  
  def build_model_logic(self):
    assert model_run_mode == skeleton_decode_mode, "serious error! not skeleton decode mode? but the model logic is sequence decode logic."
    self.model = SkeletonDecodeModel(self.type_content_data)
    
  def build_feed_dict(self, one_example):
    feed_dict = {self.skeleton_token_info_tensor:one_example[0], self.skeleton_token_info_start_tensor:one_example[1], self.skeleton_token_info_end_tensor:one_example[2]}
    return feed_dict
  
  


