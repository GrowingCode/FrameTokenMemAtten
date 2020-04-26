import tensorflow as tf
from utils.model_tensors_metrics import default_metrics_meta,\
  special_handle_metrics_meta, create_metrics_contingent_index
from metas.hyper_settings import contingent_parameters_num, num_units, top_ks
from utils.initializer import random_uniform_variable_initializer
from metas.non_hyper_constants import learning_scope, float_type


class BasicDecodeModel():
  
  '''
  the parameter start_nodes must be in the same level and be successive while
  stop_nodes do not have constraints
  '''
  
  def __init__(self, type_content_data):
    self.type_content_data = type_content_data
    self.statistical_metrics_meta = default_metrics_meta + self.create_extra_default_metrics_meta() + special_handle_metrics_meta
    self.metrics_meta = self.statistical_metrics_meta + self.create_in_use_tensors_meta()
    self.metrics_name = [metric_m[0] for metric_m in self.metrics_meta]
    self.metrics_shape = [metric_m[1] for metric_m in self.metrics_meta]
    self.index_metrics = dict((k,v) for k, v in zip(range(len(self.metrics_name)), self.metrics_name))
    self.metrics_index = {value:key for key, value in self.index_metrics.items()}
    self.metrics_contingent_index = create_metrics_contingent_index(self.metrics_meta)
    with tf.variable_scope(learning_scope):
      self.contingent_parameters = tf.get_variable("contingent", shape=[contingent_parameters_num, 2, num_units], dtype=float_type, initializer=random_uniform_variable_initializer(2, 5))
      self.contingent_parameters_for_idle = tf.get_variable("contingent2", shape=[2, 1, num_units], dtype=float_type, initializer=random_uniform_variable_initializer(20, 50))
    
  def create_extra_default_metrics_meta(self):
    return [("skeleton_loss", tf.TensorShape(())), ("skeleton_accurate", tf.TensorShape([len(top_ks)])), ("skeleton_mrr", tf.TensorShape(())), ("skeleton_count", tf.TensorShape(()))]
  
  def create_in_use_tensors_meta(self):
    result = [("token_accumulated_en", tf.TensorShape([None])), ("token_accumulated_cell", tf.TensorShape([None, num_units])), ("token_accumulated_h", tf.TensorShape([None, num_units])), ("memory_concat_cell", tf.TensorShape([None, num_units])), ("memory_concat_h", tf.TensorShape([None, num_units])), ("dup_token_accumulated_cell", tf.TensorShape([None, num_units])), ("dup_token_accumulated_h", tf.TensorShape([None, num_units])), ("dup_memory_concat_cell", tf.TensorShape([None, num_units])), ("dup_memory_concat_h", tf.TensorShape([None, num_units])), ("loop_forward_cells", tf.TensorShape([None, num_units])), ("loop_forward_hs", tf.TensorShape([None, num_units])), ("loop_backward_cells", tf.TensorShape([None, num_units])), ("loop_backward_hs", tf.TensorShape([None, num_units])), ("dup_loop_forward_cells", tf.TensorShape([None, num_units])), ("dup_loop_forward_hs", tf.TensorShape([None, num_units])), ("dup_loop_backward_cells", tf.TensorShape([None, num_units])), ("dup_loop_backward_hs", tf.TensorShape([None, num_units])), ("memory_accumulated_en", tf.TensorShape([None])), ("memory_cells", tf.TensorShape([None, num_units])), ("memory_hs", tf.TensorShape([None, num_units])), ("dup_memory_cells", tf.TensorShape([None, num_units])), ("dup_memory_hs", tf.TensorShape([None, num_units]))]
    return result
  
  
  