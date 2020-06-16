import tensorflow as tf
from utils.model_tensors_metrics import create_empty_tensorflow_tensors
from models.basic_decoder import BasicDecodeModel
from inputs.atom_embeddings import TokenAtomEmbed
from metas.hyper_settings import num_units
from utils.initializer import random_uniform_variable_initializer
from metas.non_hyper_constants import all_token_summary, TokenHitNum
from models.dup_pattern import PointerNetwork
from models.lstm import YLSTMCell
from models.lstm_procedure import one_lstm_step_and_update_memory
from models.token_sword_decode import DupTokenDecoder


class LinearDupModel(BasicDecodeModel):
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    super(LinearDupModel, self).__init__(type_content_data)
    number_of_tokens = self.type_content_data[all_token_summary][TokenHitNum]
    
    self.dup_token_lstm = YLSTMCell(125)
    self.dup_one_hot_token_embedding = tf.Variable(random_uniform_variable_initializer(25, 56, [number_of_tokens, num_units]))
    self.dup_token_embedder = TokenAtomEmbed(self.type_content_data, self.dup_one_hot_token_embedding)
    self.dup_token_pointer = PointerNetwork(222)
    
    self.dup_token_decoder = DupTokenDecoder(self.type_content_data, self.metrics_index, self.dup_token_pointer)
  
  def __call__(self, one_example, training = True):
    self.token_info_tensor = one_example[0]
    self.token_base_model_accuracy = one_example[1]
    self.token_base_model_mrr = one_example[2]
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
    token_en = self.token_info_tensor[0][i]
    token_var = self.token_info_tensor[1][i]
    token_var_relative = self.token_info_tensor[2][i]
    conserved_memory_length = self.token_info_tensor[3][i]
    token_kind = self.token_info_tensor[4][i]
    base_model_accuracy = self.token_base_model_accuracy[i]
    base_model_mrr = self.token_base_model_mrr[i]
    r_stmt_metrics_tuple = self.dup_token_decoder.decode_one_token(stmt_metrics, self.training, token_en, token_var, token_var_relative, token_kind, base_model_accuracy, base_model_mrr)
    token_metrics = list(r_stmt_metrics_tuple)
    
    token_metrics = one_lstm_step_and_update_memory("dup_", token_metrics, self.metrics_index, token_en, token_var, conserved_memory_length, self.dup_token_lstm, self.dup_token_embedder)
    
    return (i + 1, i_len, *token_metrics)
  






