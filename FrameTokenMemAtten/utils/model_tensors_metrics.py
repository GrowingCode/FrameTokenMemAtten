import tensorflow as tf
import numpy as np
from metas.non_hyper_constants import float_type, int_type
from metas.hyper_settings import top_ks


default_metrics_meta = [("all_loss", tf.TensorShape(())), ("all_accurate", tf.TensorShape([len(top_ks)])), ("all_mrr", tf.TensorShape(())), ("all_count", tf.TensorShape(())), ("sword_loss", tf.TensorShape(())), ("sword_accurate", tf.TensorShape([len(top_ks)])), ("sword_mrr", tf.TensorShape(())), ("sword_dup_loss", tf.TensorShape(())), ("sword_dup_accurate", tf.TensorShape([len(top_ks)])), ("sword_dup_mrr", tf.TensorShape(())), ("sword_lm_loss", tf.TensorShape(())), ("sword_lm_accurate", tf.TensorShape([len(top_ks)])), ("sword_lm_mrr", tf.TensorShape(())), ("sword_count", tf.TensorShape(())), ("token_loss", tf.TensorShape(())), ("token_accurate", tf.TensorShape([len(top_ks)])), ("token_mrr", tf.TensorShape(())), ("token_lm_loss", tf.TensorShape(())), ("token_lm_accurate", tf.TensorShape([len(top_ks)])), ("token_lm_mrr", tf.TensorShape(())), ("token_dup_loss", tf.TensorShape(())), ("token_dup_accurate", tf.TensorShape([len(top_ks)])), ("token_dup_mrr", tf.TensorShape(())), ("token_sword_beam_accurate", tf.TensorShape([len(top_ks)])), ("token_sword_beam_mrr", tf.TensorShape(())), ("token_count", tf.TensorShape(()))]

def create_empty_tensorflow_tensors(metrics_meta, contingent_parameters, metrics_contingent_index):
  result = []
  for m_name in metrics_meta:
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
    else:
      if m_name.endswith("accumulated_cell") or m_name.endswith("accumulated_h"):
        ct = contingent_parameters[metrics_contingent_index[m_name]][:,:]
      else:
        ct = contingent_parameters[metrics_contingent_index[m_name]][-2:-1,:]
    result.append(ct)
  return tuple(result)

def create_empty_numpy_metrics(self, metrics_meta):
  result = []
  for m_name in metrics_meta:
    if m_name.endswith("_loss"):
      ct = 0.0
    elif m_name.endswith("_accurate"):
      ct = np.zeros([len(top_ks)], float_type)
    elif m_name.endswith("_mrr"):
      ct = 0.0
    elif m_name.endswith("_count"):
      ct = 0
    result.append(ct)
  return tuple(result)

def create_metrics_contingent_index(self, metrics_meta):
  self.metrics_contingent = {}
  idx = 0
  for m_name in metrics_meta:
    if m_name.endswith("_cell") or m_name.endswith("_h") or m_name.endswith("_cells") or m_name.endswith("_hs"):
      self.metrics_contingent[m_name] = idx
      idx = idx+1


