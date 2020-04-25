from models.model_runner import ModelRunner
import tensorflow as tf
from metas.non_hyper_constants import int_type
from metas.hyper_settings import model_run_mode, sequence_decode_mode
from models.sequence.sequence_decoder import SequenceDecodeModel


class SequenceModelRunner(ModelRunner):
  
  def __init__(self, sess):
    super(SequenceModelRunner, self).__init__(sess)
  
  def build_input_place_holder(self):
    self.token_info_tensor = tf.compat.v1.placeholder(int_type, [None])
    return [self.token_info_tensor]
  
  def build_model_logic(self):
    assert model_run_mode == sequence_decode_mode, "serious error! not sequence decode mode? but the model logic is sequence decode logic."
    self.model = SequenceDecodeModel(self.type_content_data)
    
  def build_feed_dict(self, one_example):
    token_info_tensor = one_example[0]
    feed_dict = {self.token_info_tensor : token_info_tensor}
    return feed_dict
  
  


