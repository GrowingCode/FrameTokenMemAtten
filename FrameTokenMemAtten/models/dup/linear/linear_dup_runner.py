from models.model_runner import ModelRunner
import tensorflow as tf
from metas.non_hyper_constants import int_type, float_type
from metas.hyper_settings import model_run_mode, top_ks,\
  linear_dup_mode
from inputs.example_data_loader import build_linear_dup_feed_dict
from models.dup.linear.linear_dup_decoder import LinearDupModel


class LinearDupRunner(ModelRunner):
  
  def __init__(self, sess):
    super(LinearDupRunner, self).__init__(sess)
  
  def set_up_example_loader(self):
    self.example_loader = build_linear_dup_feed_dict
  
  def build_input_place_holder(self):
    self.token_info_tensor = tf.compat.v1.placeholder(int_type, [None, None])
    self.token_base_model_accuracy = tf.compat.v1.placeholder(float_type, [None, len(top_ks)])
    self.token_base_model_mrr = tf.compat.v1.placeholder(float_type, [None])
    return [self.token_info_tensor, self.token_base_model_accuracy, self.token_base_model_mrr]
  
  def build_model_logic(self):
    assert model_run_mode == linear_dup_mode, "serious error! not sequence decode mode? but the model logic is linear dup logic."
    self.model = LinearDupModel(self.type_content_data)
    
  def build_feed_dict(self, one_example):
    token_info_tensor = one_example[0][0]
    token_base_model_accuracy = one_example[1]
    token_base_model_mrr = one_example[2]
    feed_dict = {self.token_info_tensor : token_info_tensor, self.token_base_model_accuracy : token_base_model_accuracy, self.token_base_model_mrr : token_base_model_mrr}
    return feed_dict
  
  


