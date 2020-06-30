from metas.hyper_settings import only_memory_mode, token_memory_mode,\
  concat_memory_mode, num_units,\
  no_memory_mode, abs_memory_size, abs_size_var_novar_all_concat_memory_mode,\
  abs_size_concat_memory_mode, compute_memory_in_only_memory_mode
from models.mem import update_one_variable, compute_integrated_memory
import tensorflow as tf
from metas.non_hyper_constants import int_type


def one_lstm_step_and_update_memory(prefix, token_metrics, metrics_index, token_en, token_var, conserved_memory_length, token_lstm, token_embedder, integrate_computer=None):
  dup_cell = token_metrics[metrics_index[prefix + "token_cell"]]
  dup_h = token_metrics[metrics_index[prefix + "token_h"]]
  
  if token_memory_mode == no_memory_mode:
    pass
  elif token_memory_mode == only_memory_mode:
    dup_acc_cells = token_metrics[metrics_index[prefix + "memory_acc_cell"]]
    dup_acc_hs = token_metrics[metrics_index[prefix + "memory_acc_h"]]
    dup_acc_ens = token_metrics[metrics_index[prefix + "memory_en"]]
    
    n_dup_cell, n_dup_h = dup_cell, dup_h
    if compute_memory_in_only_memory_mode:
      assert integrate_computer != None
      n_dup_cell, n_dup_h = compute_integrated_memory(integrate_computer, token_var, dup_cell, dup_h, dup_acc_cells, dup_acc_hs)
    
    token_metrics[metrics_index[prefix + "memory_en"]], token_metrics[metrics_index[prefix + "memory_acc_cell"]], token_metrics[metrics_index[prefix + "memory_acc_h"]] = update_one_variable(token_var, token_en, n_dup_cell, n_dup_h, dup_acc_ens, dup_acc_cells, dup_acc_hs)
  else:
    if token_memory_mode == concat_memory_mode:
      to_concat = tf.cast(token_var > 0, int_type)
      r_memory_length = conserved_memory_length
    elif token_memory_mode == abs_size_concat_memory_mode:
      to_concat = tf.cast(token_var > 0, int_type)
      r_memory_length = abs_memory_size
    elif token_memory_mode == abs_size_var_novar_all_concat_memory_mode:
      to_concat = tf.constant(1, int_type)
      r_memory_length = abs_memory_size
    
    concat_en = [token_en]
    concat_cell = dup_cell
    concat_h = dup_h
    concat_en = tf.slice(concat_en, [1-to_concat], [to_concat])
    concat_cell = tf.slice(concat_cell, [1-to_concat, 0], [to_concat, num_units])
    concat_h = tf.slice(concat_h, [1-to_concat, 0], [to_concat, num_units])
    
    token_metrics[metrics_index[prefix + "memory_en"]] = tf.concat([token_metrics[metrics_index[prefix + "memory_en"]], concat_en], axis=0)
    token_metrics[metrics_index[prefix + "memory_acc_cell"]] = tf.concat([token_metrics[metrics_index[prefix + "memory_acc_cell"]], concat_cell], axis=0)
    token_metrics[metrics_index[prefix + "memory_acc_h"]] = tf.concat([token_metrics[metrics_index[prefix + "memory_acc_h"]], concat_h], axis=0)
    
    curr_mem_len = tf.shape(token_metrics[metrics_index[prefix + "memory_en"]])[0]
    slice_start = curr_mem_len - r_memory_length
    r_slice_start = tf.cast(tf.logical_and(slice_start > 0, slice_start < curr_mem_len), int_type) * slice_start
    r_slice_length = curr_mem_len - r_slice_start
    
    token_metrics[metrics_index[prefix + "memory_en"]] = tf.slice(token_metrics[metrics_index[prefix + "memory_en"]], [r_slice_start], [r_slice_length])
    token_metrics[metrics_index[prefix + "memory_acc_cell"]] = tf.slice(token_metrics[metrics_index[prefix + "memory_acc_cell"]], [r_slice_start, 0], [r_slice_length, num_units])
    token_metrics[metrics_index[prefix + "memory_acc_h"]] = tf.slice(token_metrics[metrics_index[prefix + "memory_acc_h"]], [r_slice_start, 0], [r_slice_length, num_units])
    
  _, (new_dup_cell, new_dup_h) = token_lstm(token_embedder.compute_h(token_en), (dup_cell, dup_h))
  token_metrics[metrics_index[prefix + "token_cell"]] = new_dup_cell
  token_metrics[metrics_index[prefix + "token_h"]] = new_dup_h
  return token_metrics


def one_lstm_step(prefix, token_metrics, metrics_index, token_en, token_lstm, token_embedder):
  dup_cell = token_metrics[metrics_index[prefix + "token_cell"]]
  dup_h = token_metrics[metrics_index[prefix + "token_h"]]
  _, (new_dup_cell, new_dup_h) = token_lstm(token_embedder.compute_h(token_en), (dup_cell, dup_h))
  token_metrics[metrics_index[prefix + "token_cell"]] = new_dup_cell
  token_metrics[metrics_index[prefix + "token_h"]] = new_dup_h
  return token_metrics


def backward_varied_lstm_steps(inputs, ini_cell, ini_h, t_lstm):
  
  def loop_cond(i_start, i_end, *_):
    return tf.greater_equal(i_start, i_end)
  
  def loop_body(i_start, i_end, b_cell, b_h):
    _, (b_cell, b_h) = t_lstm(tf.expand_dims(inputs[i_start], axis=0), (b_cell, b_h))
    return i_start - 1, i_end, b_cell, b_h
  
  s_start = tf.shape(inputs)[0] - 1
  s_end = tf.constant(0, int_type)
  _, _, b_cell, b_h = tf.while_loop(loop_cond, loop_body, [s_start, s_end, ini_cell, ini_h])
  return b_h, (b_cell, b_h)











