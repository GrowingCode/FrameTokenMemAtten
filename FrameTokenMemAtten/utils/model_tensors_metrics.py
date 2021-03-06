import tensorflow as tf
from metas.non_hyper_constants import float_type, int_type
from metas.hyper_settings import top_ks
from utils.tensor_array_stand import convert_tensor_array_to_lists_of_tensors,\
  filter_out_invalid_mark


default_metrics_meta = [("all_loss", tf.TensorShape(())), ("all_accurate", tf.TensorShape([len(top_ks)])), ("all_mrr", tf.TensorShape(())), ("all_count", tf.TensorShape(())), ("sword_loss", tf.TensorShape(())), ("sword_accurate", tf.TensorShape([len(top_ks)])), ("sword_mrr", tf.TensorShape(())), ("sword_dup_loss", tf.TensorShape(())), ("sword_dup_accurate", tf.TensorShape([len(top_ks)])), ("sword_dup_mrr", tf.TensorShape(())), ("sword_lm_loss", tf.TensorShape(())), ("sword_lm_accurate", tf.TensorShape([len(top_ks)])), ("sword_lm_mrr", tf.TensorShape(())), ("sword_count", tf.TensorShape(())), ("token_loss", tf.TensorShape(())), ("token_accurate", tf.TensorShape([len(top_ks)])), ("token_mrr", tf.TensorShape(())), ("token_lm_loss", tf.TensorShape(())), ("token_lm_accurate", tf.TensorShape([len(top_ks)])), ("token_lm_mrr", tf.TensorShape(())), ("token_dup_loss", tf.TensorShape(())), ("token_dup_accurate", tf.TensorShape([len(top_ks)])), ("token_dup_mrr", tf.TensorShape(())), ("token_count", tf.TensorShape(()))]
special_handle_metrics_meta = [("atom_int_noavg", tf.TensorShape(None)), ("token_accurate_each_noavg", tf.TensorShape(None)), ("token_mrr_each_noavg", tf.TensorShape(None)), ("token_count_each_int_noavg", tf.TensorShape(None))]

def create_empty_tensorflow_tensors(metrics_meta, contingent_parameters, metrics_contingent_index):
  result = []
  for mm in metrics_meta:
    m_name = mm[0]
    if m_name.endswith("_en"):
      ct = tf.zeros([1], int_type)
    elif m_name.endswith("_loss"):
      ct = tf.constant(0.0, float_type)
    elif m_name.endswith("_accurate"):
      ct = tf.zeros([len(top_ks)], float_type)
    elif m_name.endswith("_mrr"):
      ct = tf.constant(0.0, float_type)
    elif m_name.endswith("_count"):
      ct = tf.constant(0, int_type)
    elif m_name.endswith("_int_noavg"):
      ct = tf.TensorArray(int_type, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False)
    elif m_name.endswith("_noavg"):
      ct = tf.TensorArray(float_type, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False)
    else:
#       if m_name.endswith("accumulated_cell") or m_name.endswith("accumulated_h"):
#         ct = contingent_parameters[metrics_contingent_index[m_name]][:,:]
#       else:
      m_idx = metrics_contingent_index[m_name]
#       print("m_name:" + m_name + "#m_idx:" + str(m_idx))
#       print(contingent_parameters)
#       p_op = tf.print(['m_idx', m_idx, 'shape_of_contingent_parameters', tf.shape(contingent_parameters)])
#       with tf.control_dependencies([p_op]):
      ct = contingent_parameters[m_idx]#[-2:-1,:]
    result.append(ct)
  return tuple(result)


def ensure_tensor_array_to_tensor_list_in_metrics(metrics, metrics_meta, metrics_index):
  for mm in metrics_meta:
    m_name = mm[0]
    if m_name.endswith("_noavg"):
      metrics[metrics_index[m_name]] = convert_tensor_array_to_lists_of_tensors(metrics[metrics_index[m_name]])
      

def filter_out_invalid_mark_in_metrics(metrics, metrics_meta, metrics_index):
  for mm in metrics_meta:
    m_name = mm[0]
    if m_name.endswith("_noavg"):
      metrics[metrics_index[m_name]] = filter_out_invalid_mark(metrics[metrics_index[m_name]])


def create_metrics_contingent_index(metrics_meta):
  metrics_contingent = {}
  idx = 0
  for mm in metrics_meta:
    m_name = mm[0]
    if m_name.endswith("_cell") or m_name.endswith("_h") or m_name.endswith("_cells") or m_name.endswith("_hs"):
      metrics_contingent[m_name] = idx
      idx = idx+1
  return metrics_contingent



