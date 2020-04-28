import tensorflow as tf
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from models.token_sword_decode import decode_one_token
from models.basic_decoder import BasicDecodeModel
from inputs.atom_embeddings import TokenAtomEmbed
from metas.hyper_settings import num_units, decode_attention_way,\
  decode_no_attention
from utils.initializer import random_uniform_variable_initializer
from metas.non_hyper_constants import all_token_summary, TokenHitNum,\
  float_type, lstm_initialize_range
from models.attention import YAttention
from tensorflow_core.python.ops.rnn_cell_impl import LSTMCell


class SequenceDecodeModel(BasicDecodeModel):
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    super(SequenceDecodeModel, self).__init__(type_content_data)
    self.token_lstm = LSTMCell(num_units, initializer=tf.compat.v1.random_uniform_initializer(-lstm_initialize_range, lstm_initialize_range, seed=100, dtype=float_type), forget_bias=0.0, dtype=float_type)
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    self.linear_token_output_w = tf.Variable(random_uniform_variable_initializer(256, 566, [number_of_tokens, num_units]))
    self.one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(256, 56, [number_of_tokens, num_units]))
    self.token_embedder = TokenAtomEmbed(self.type_content_data, self.one_hot_token_embedding)
    
    self.token_attention = None
    if decode_attention_way > decode_no_attention:
      self.token_attention = YAttention(10)
    
    self.dup_token_embedder, self.dup_token_lstm, self.token_pointer = None, None, None
  
  def __call__(self, one_example, training = True):
    self.token_info_tensor = one_example[0]
    self.training = training
    ini_metrics = list(create_empty_tensorflow_tensors(self.metrics_meta, self.contingent_parameters, self.metrics_contingent_index))
    sequence_length = tf.shape(self.token_info_tensor)[-1]
    end_index = sequence_length - 1
#     p_op = tf.print(["sequence_length:", sequence_length])
#     with tf.control_dependencies([p_op]):
    f_res = tf.while_loop(self.token_iterate_cond, self.token_iterate_body, [0, end_index, *ini_metrics], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), *self.metrics_shape], parallel_iterations=1)
    f_res = list(f_res[2:2+len(self.statistical_metrics_meta)])
    return f_res
  
  def token_iterate_cond(self, i, i_len, *_):
    return tf.less_equal(i, i_len)
  
  def token_iterate_body(self, i, i_len, *stmt_metrics_tuple):
    stmt_metrics = list(stmt_metrics_tuple)
    oracle_type_content_en = self.token_info_tensor[0][i]
    r_stmt_metrics_tuple = decode_one_token(self.type_content_data, self.training, oracle_type_content_en, -1, -1, self.metrics_index, stmt_metrics, self.linear_token_output_w, self.token_lstm, self.token_embedder, self.token_attention)
    return (i + 1, i_len, *r_stmt_metrics_tuple)
  






