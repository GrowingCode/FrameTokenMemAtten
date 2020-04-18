import tensorflow as tf
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from models.token_sword_decode import decode_one_token
from models.basic_decoder import BasicDecodeModel
from inputs.atom_embeddings import TokenAtomEmbed
from metas.hyper_settings import num_units
from utils.initializer import random_uniform_variable_initializer
from metas.non_hyper_constants import all_token_summary, TokenHitNum
from models.lstm import YLSTMCell


class SequenceDecodeModel(BasicDecodeModel):
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    super(SequenceDecodeModel, self).__init__(type_content_data)
    self.token_lstm = YLSTMCell()
    
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    self.linear_token_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_tokens, num_units]))
    self.one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_tokens, num_units]))
    self.token_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_embedding)
    
    self.dup_token_embedder, self.dup_token_lstm, self.token_pointer = None, None, None
  
  def __call__(self, token_info_tensor, token_info_start_tensor, token_info_end_tensor, training = True):
    assert token_info_start_tensor != None and token_info_end_tensor != None
    self.token_info_tensor = token_info_tensor
    self.training = training
    ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    f_res = tf.while_loop(self.token_iterate_cond, self.token_iterate_body, [0, tf.shape(self.token_info_tensor)[-1], *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2+len(self.statistical_metrics_meta)])
    return f_res
  
  def token_iterate_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_iterate_body(self, i, i_len, ini_i, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
    r_stmt_metrics_tuple = decode_one_token(self.type_content_data, self.training, oracle_type_content_en, -1, -1, self.metrics_index, stmt_metrics, self.linear_token_output_w, self.token_lstm, self.token_embedder)
    return (i + 1, i_len, ini_i, *r_stmt_metrics_tuple)
  






