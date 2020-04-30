from metas.hyper_settings import only_memory_mode, token_memory_mode,\
  concat_memory_mode, num_units
from models.mem import update_one_variable
import tensorflow as tf
from metas.non_hyper_constants import int_type, float_type


def one_lstm_step_and_update_memory(prefix, token_metrics, metrics_index, token_en, token_var, conserved_memory_length, token_lstm, token_embedder):
  dup_cell = token_metrics[metrics_index[prefix + "token_cell"]]
  dup_h = token_metrics[metrics_index[prefix + "token_h"]]
  
  if token_memory_mode == only_memory_mode:
    dup_acc_cells = token_metrics[metrics_index[prefix + "memory_acc_cell"]]
    dup_acc_hs = token_metrics[metrics_index[prefix + "memory_acc_h"]]
    dup_acc_ens = token_metrics[metrics_index[prefix + "memory_en"]]
    token_metrics[metrics_index[prefix + "memory_en"]], token_metrics[metrics_index[prefix + "memory_acc_cell"]], token_metrics[metrics_index[prefix + "memory_acc_h"]] = update_one_variable(token_var, token_en, dup_cell, dup_h, dup_acc_ens, dup_acc_cells, dup_acc_hs)
  elif token_memory_mode == concat_memory_mode:
    to_concat = tf.cast(token_var > 0, int_type)
    concat_en = tf.stack([tf.zeros([0], int_type), [token_en]])[to_concat]
    concat_cell = tf.stack([tf.zeros([0, num_units], float_type), dup_cell])[to_concat]
    concat_h = tf.stack([tf.zeros([0, num_units], float_type), dup_h])[to_concat]
    
    token_metrics[metrics_index[prefix + "memory_en"]] = tf.concat([token_metrics[metrics_index[prefix + "memory_en"]], concat_en], axis=0)
    token_metrics[metrics_index[prefix + "memory_acc_cell"]] = tf.concat([token_metrics[metrics_index[prefix + "memory_acc_cell"]], concat_cell], axis=0)
    token_metrics[metrics_index[prefix + "memory_acc_h"]] = tf.concat([token_metrics[metrics_index[prefix + "memory_acc_h"]], concat_h], axis=0)
    
    curr_mem_len = tf.shape(token_metrics[metrics_index[prefix + "memory_en"]])[0]
    slice_start = curr_mem_len - conserved_memory_length
    r_slice_start = tf.cast(tf.logical_and(slice_start > 0, slice_start < curr_mem_len), int_type) * slice_start
    
    token_metrics[metrics_index[prefix + "memory_en"]] = token_metrics[metrics_index[prefix + "memory_en"]][r_slice_start:]
    token_metrics[metrics_index[prefix + "memory_acc_cell"]] = token_metrics[metrics_index[prefix + "memory_acc_cell"]][r_slice_start:, :]
    token_metrics[metrics_index[prefix + "memory_acc_h"]] = token_metrics[metrics_index[prefix + "memory_acc_h"]][r_slice_start:, :]
    
  _, (new_dup_cell, new_dup_h) = token_lstm(token_embedder.compute_h(token_en), (dup_cell, dup_h))
  token_metrics[metrics_index["dup_token_cell"]] = new_dup_cell
  token_metrics[metrics_index["dup_token_h"]] = new_dup_h
  return token_metrics






