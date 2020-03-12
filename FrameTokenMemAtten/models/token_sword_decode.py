from inputs.atom_embeddings import sword_sequence_for_token
from metas.hyper_settings import use_dup_model, compute_token_memory, \
  accumulated_token_max_length, num_units, top_ks
from metas.non_hyper_constants import int_type, bool_type, float_type
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings,\
  compute_logits_given_to_deocde_embed_with_computed_embeddings
import tensorflow as tf
from utils.tensor_concat import concat_in_fixed_length_two_dimension, \
  concat_in_fixed_length_one_dimension


def decode_one_token(training, oracle_type_content_en, oracle_type_content_var, oracle_type_content_var_relative, metrics_index, token_metrics, linear_token_output_w, token_lstm, token_embedder, dup_token_lstm=None, dup_token_embedder=None, token_pointer=None):
  ''' typical token swords prediction '''
  cell = tf.convert_to_tensor([token_metrics[metrics_index["token_accumulated_cell"]][-1]])
  h = tf.convert_to_tensor([token_metrics[metrics_index["token_accumulated_h"]][-1]])
  ''' for attention use '''
  before_accurate = token_metrics[metrics_index["all_accurate"]]
  before_mrr = token_metrics[metrics_index["all_mrr"]]
  before_token_accurate = token_metrics[metrics_index["token_accurate"]]
  before_token_mrr = token_metrics[metrics_index["token_mrr"]]
  ''' decode and compute sword level accurate '''
  mrr_of_this_node, accurate_of_this_node, loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(training, linear_token_output_w, oracle_type_content_en, h)
  token_metrics[metrics_index["token_loss"]] = token_metrics[metrics_index["token_loss"]] + loss_of_this_node
  token_metrics[metrics_index["token_accurate"]] = token_metrics[metrics_index["token_accurate"]] + accurate_of_this_node
  token_metrics[metrics_index["token_mrr"]] = token_metrics[metrics_index["token_mrr"]] + mrr_of_this_node
  token_metrics[metrics_index["token_lm_loss"]] = token_metrics[metrics_index["token_lm_loss"]] + loss_of_this_node
  token_metrics[metrics_index["token_lm_accurate"]] = token_metrics[metrics_index["token_lm_accurate"]] + accurate_of_this_node
  token_metrics[metrics_index["token_lm_mrr"]] = token_metrics[metrics_index["token_lm_mrr"]] + mrr_of_this_node
  token_metrics[metrics_index["token_count"]] = token_metrics[metrics_index["token_count"]] + 1
  token_metrics[metrics_index["all_loss"]] = token_metrics[metrics_index["all_loss"]] + loss_of_this_node
  token_metrics[metrics_index["all_accurate"]] = token_metrics[metrics_index["all_accurate"]] + accurate_of_this_node
  token_metrics[metrics_index["all_mrr"]] = token_metrics[metrics_index["all_mrr"]] + mrr_of_this_node
  token_metrics[metrics_index["all_count"]] = token_metrics[metrics_index["all_count"]] + 1
  
  after_accurate = token_metrics[metrics_index["all_accurate"]]
  after_mrr = token_metrics[metrics_index["all_mrr"]]
  after_token_accurate = token_metrics[metrics_index["token_accurate"]]
  after_token_mrr = token_metrics[metrics_index["token_mrr"]]
  
  ''' dup token prediction '''
  if use_dup_model:
    ''' compute duplicate pattern '''
    dup_cell = tf.convert_to_tensor([token_metrics[metrics_index["dup_token_accumulated_cell"]][-1]])
    dup_h = tf.convert_to_tensor([token_metrics[metrics_index["dup_token_accumulated_h"]][-1]])
    dup_n_size = tf.shape(token_metrics[metrics_index["dup_token_accumulated_h"]])[0]
    dup_acc_hs = tf.slice(token_metrics[metrics_index["dup_token_accumulated_h"]], [0, 0], [dup_n_size - 1, num_units])
    dup_acc_ens = token_metrics[metrics_index["token_accumulated_en"]]  # tf.slice(, [0], [dup_n_size - 1])
    
    r_var_relative = oracle_type_content_var_relative
    
    if compute_token_memory:
      dup_acc_hs = token_metrics[metrics_index["dup_memory_hs"]]
      dup_acc_ens = token_metrics[metrics_index["memory_accumulated_en"]]
      r_var_relative = tf.shape(dup_acc_ens)[-1] - oracle_type_content_var
    
    r_var_relative_valid = tf.cast(tf.greater(r_var_relative, 0), int_type)
    a_r_op = tf.Assert(tf.logical_or(tf.cast(1-r_var_relative_valid, bool_type), tf.equal(dup_acc_ens[tf.shape(dup_acc_ens)[0]-r_var_relative*r_var_relative_valid-1+r_var_relative_valid], oracle_type_content_en)), ["invalid variable relative"])
    a_op = tf.assert_equal(tf.shape(dup_acc_ens)[0], tf.shape(dup_acc_hs)[0])
    with tf.control_dependencies([a_r_op, a_op]):
      dup_logits, dup_max_arg_acc_h = token_pointer.compute_logits(dup_acc_hs, dup_h)
      is_dup_logits = token_pointer.compute_is_dup_logits(dup_max_arg_acc_h, dup_h)
      dup_mrr_of_this_node, dup_accurate_of_this_node, dup_loss_of_this_node, r_dup_mrr, r_dup_accurate, predict_to_use_pre_exist = token_pointer.compute_dup_loss(training, dup_acc_ens, oracle_type_content_en, r_var_relative, is_dup_logits, dup_logits)
    
    token_metrics[metrics_index["token_dup_loss"]] = token_metrics[metrics_index["token_dup_loss"]] + dup_loss_of_this_node
    token_metrics[metrics_index["token_dup_accurate"]] = token_metrics[metrics_index["token_dup_accurate"]] + dup_accurate_of_this_node
    token_metrics[metrics_index["token_dup_mrr"]] = token_metrics[metrics_index["token_dup_mrr"]] + dup_mrr_of_this_node
    token_metrics[metrics_index["all_loss"]] = token_metrics[metrics_index["all_loss"]] + dup_loss_of_this_node
    ''' compute accurate '''
    token_metrics[metrics_index["all_accurate"]] = before_accurate + tf.stack([after_accurate - before_accurate, r_dup_accurate], axis=0)[predict_to_use_pre_exist]
    token_metrics[metrics_index["all_mrr"]] = before_mrr + tf.stack([after_mrr - before_mrr, r_dup_mrr], axis=0)[predict_to_use_pre_exist]
    token_metrics[metrics_index["token_accurate"]] = before_token_accurate + tf.stack([after_token_accurate - before_token_accurate, r_dup_accurate], axis=0)[predict_to_use_pre_exist]
    token_metrics[metrics_index["token_mrr"]] = before_token_mrr + tf.stack([after_token_mrr - before_token_mrr, r_dup_mrr], axis=0)[predict_to_use_pre_exist]
    
    new_dup_token_cell, new_dup_token_h = dup_token_lstm(dup_token_embedder.compute_h(oracle_type_content_en), dup_cell, dup_h)
    token_metrics[metrics_index["dup_token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["dup_token_accumulated_cell"]], new_dup_token_cell, accumulated_token_max_length+1)
    token_metrics[metrics_index["dup_token_accumulated_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["dup_token_accumulated_h"]], new_dup_token_h, accumulated_token_max_length+1)
    
  token_metrics[metrics_index["token_accumulated_en"]] = concat_in_fixed_length_one_dimension(token_metrics[metrics_index["token_accumulated_en"]], [oracle_type_content_en], accumulated_token_max_length)
  ''' predict next node '''
  new_cell, new_h = token_lstm(token_embedder.compute_h(oracle_type_content_en), cell, h)
  token_metrics[metrics_index["token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_cell"]], new_cell, accumulated_token_max_length+1)
  token_metrics[metrics_index["token_accumulated_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_h"]], new_h, accumulated_token_max_length+1)
  return tuple(token_metrics)


def decode_swords_of_one_token(training, type_content_data, token_en, metrics_index, metrics_shape, token_metrics, token_lstm, token_embedder, linear_sword_output_w, sword_embedder, sword_lstm):
  token_atom_sequence = sword_sequence_for_token(type_content_data, token_en)
  atom_length_in_float = tf.cast(tf.shape(token_atom_sequence)[-1], float_type)
  
  def decode_one_sword_cond(w, w_len, *_):
    return tf.less(w, w_len)
   
  def decode_one_sword_body(w, w_len, sword_c, sword_h, *sword_metrics_tuple):
    sword_metrics = list(sword_metrics_tuple)
    oracle_sword_en = token_atom_sequence[w]
    mrr_of_this_node, accurate_of_this_node, loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(training, linear_sword_output_w, oracle_sword_en, sword_h)
    ''' predict next node '''
    e_embed = sword_embedder.compute_h(oracle_sword_en)
    new_sword_c, new_sword_h = sword_lstm(e_embed, sword_c, sword_h)
    ''' final loss and accurate '''
    sword_metrics[metrics_index["sword_loss"]] = sword_metrics[metrics_index["sword_loss"]] + loss_of_this_node  # + dup_loss_of_this_node
    sword_metrics[metrics_index["sword_accurate"]] = sword_metrics[metrics_index["sword_accurate"]] + accurate_of_this_node
    sword_metrics[metrics_index["sword_mrr"]] = sword_metrics[metrics_index["sword_mrr"]] + mrr_of_this_node
    sword_metrics[metrics_index["sword_count"]] = sword_metrics[metrics_index["sword_count"]] + 1
    
    sword_metrics[metrics_index["all_loss"]] = sword_metrics[metrics_index["all_loss"]] + loss_of_this_node / atom_length_in_float
    sword_metrics[metrics_index["all_accurate"]] = sword_metrics[metrics_index["all_accurate"]] + accurate_of_this_node / atom_length_in_float
    sword_metrics[metrics_index["all_mrr"]] = sword_metrics[metrics_index["all_mrr"]] + mrr_of_this_node / atom_length_in_float
    return (w + 1, w_len, new_sword_c, new_sword_h, *sword_metrics)
  
  cell = tf.convert_to_tensor([token_metrics[metrics_index["token_accumulated_cell"]][-1]])
  h = tf.convert_to_tensor([token_metrics[metrics_index["token_accumulated_h"]][-1]])
  
  if training:
    beam_mrr_of_this_node, beam_accurate_of_this_node = tf.constant(0.0, float_type), tf.zeros([len(top_ks)], float_type)
  else:
    beam_mrr_of_this_node, beam_accurate_of_this_node = compute_atom_sequence_beam_accurate(linear_sword_output_w, sword_lstm, sword_embedder, cell, h, token_atom_sequence)
  
  new_cell, new_h = token_lstm(token_embedder.compute_h(token_en), cell, h)
  token_metrics[metrics_index["token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_cell"]], new_cell, accumulated_token_max_length)
  token_metrics[metrics_index["token_accumulated_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_h"]], new_h, accumulated_token_max_length)
  
  token_metrics[metrics_index["token_sword_beam_accurate"]] = token_metrics[metrics_index["token_sword_beam_accurate"]] + beam_accurate_of_this_node
  token_metrics[metrics_index["token_sword_beam_mrr"]] = token_metrics[metrics_index["token_sword_beam_mrr"]] + beam_mrr_of_this_node
  token_metrics_res = tf.while_loop(decode_one_sword_cond, decode_one_sword_body, [0, tf.shape(token_atom_sequence)[-1], cell, h, *token_metrics], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([1, num_units]), *metrics_shape], parallel_iterations=1)
  r_token_metrics = list(token_metrics_res[4:])
  r_token_metrics[metrics_index["token_count"]] = r_token_metrics[metrics_index["token_count"]] + 1
  r_token_metrics[metrics_index["all_count"]] = r_token_metrics[metrics_index["all_count"]] + 1
  return r_token_metrics


def compute_atom_sequence_beam_accurate(linear_sword_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, oracle_atoms):
  atom_length = tf.shape(oracle_atoms)[-1]
  computed_ens = compute_beam_sequences(linear_sword_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, atom_length)
  mrr, accurate = compute_beam_accurates(computed_ens, oracle_atoms)
  return mrr, accurate


def compute_beam_sequences(linear_atom_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, atom_length):
    
  def atom_decode_one(top_n, cell, h, prob_scalar, computed_en): 
    n_cells = tf.zeros([0, num_units], float_type)
    n_hs = tf.zeros([0, num_units], float_type)
    logits = compute_logits_given_to_deocde_embed_with_computed_embeddings(linear_atom_output_w, h)
    probs = tf.nn.softmax(logits)
    probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    _, n_atom_ens = tf.nn.top_k(probs, top_n)
    n_probs = tf.gather(probs, n_atom_ens) + prob_scalar
    n_computed_ens = tf.zeros([0, tf.shape(computed_en)[-1] + 1], int_type)
    for _ in range(top_n):
      n_cell, n_h = sword_lstm(sword_embedder.compute_sword_h(n_atom_ens[i]), cell, h)
      n_cells = tf.concat([n_cells, n_cell], axis=0)
      n_hs = tf.concat([n_hs, n_h], axis=0)
      n_computed_en = tf.concat([computed_en, [n_atom_ens[i]]], axis=0)
      n_computed_ens = tf.concat([n_computed_ens, [n_computed_en]], axis=0)
    return n_cells, n_hs, n_probs, n_computed_ens  # n_accurates, 
  
  def atom_sequence_decoding_cond(i, i_len, *_):
    return tf.less(i, i_len)

  ''' body '''

  def atom_sequence_decoding_body(i, i_len, cells, hs, probs, computed_ens):  # accurates
    beam_size = top_ks[-1]
    ''' use h to decode '''
    ''' next char embedding '''
    
    def atom_decode_filter(cells, hs, probs, atom_ens, retain_size):
      r_size = tf.minimum(tf.shape(probs)[0], retain_size)
      _, indices = tf.nn.top_k(probs, r_size)
      return tf.gather(cells, indices), tf.gather(hs, indices), tf.gather(probs, indices), tf.gather(atom_ens, indices)  # , tf.gather(accurates, indices)
  
    def beam_cond(j, j_len, *_):
      return tf.less(j, j_len)
    
    def beam_body(j, j_len, candi_cells, candi_hs, candi_atom_probs, candi_atom_ens):  # candi_accurates, 
      n_cells, n_hs, n_probs, n_computed_ens = atom_decode_one(beam_size, [cells[j]], [hs[j]], probs[j], computed_ens[j])  # , computed_ens[j][-1], n_accurates, accurates[j], oracle_atoms[i]
      candi_cells = tf.concat([candi_cells, n_cells], axis=0)
      candi_hs = tf.concat([candi_hs, n_hs], axis=0)
      candi_atom_probs = tf.concat([candi_atom_probs, n_probs], axis=0)
      candi_atom_ens = tf.concat([candi_atom_ens, n_computed_ens], axis=0)
      return j + 1, j_len, candi_cells, candi_hs, candi_atom_probs, candi_atom_ens  # candi_accurates, 
    
    j = tf.constant(0, int_type)
    j_len = tf.shape(cells)[0]
    candi_cells = tf.zeros([0, num_units], float_type)
    candi_hs = tf.zeros([0, num_units], float_type)
    candi_probs = tf.zeros([0], float_type)
    candi_atom_ens = tf.zeros([0, tf.shape(computed_ens)[-1] + 1], int_type)
    _, _, candi_cells, candi_hs, candi_probs, candi_atom_ens = tf.while_loop(beam_cond, beam_body, [j, j_len, candi_cells, candi_hs, candi_probs, candi_atom_ens], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None]), tf.TensorShape([None, None])], parallel_iterations=1)
    candi_cells, candi_hs, candi_probs, candi_atom_ens = atom_decode_filter(candi_cells, candi_hs, candi_probs, candi_atom_ens, beam_size)
    return i + 1, i_len, candi_cells, candi_hs, candi_probs, candi_atom_ens  # , candi_accurates
  
  ''' compute loss '''
  i = tf.constant(0, int_type)
  i_len = atom_length  # tf.shape(oracle_atoms)[-1]
  cells = begin_cell
  hs = begin_h
  computed_ens = tf.zeros([1, 0], int_type)
  _, _, _, _, computed_ens = tf.while_loop(atom_sequence_decoding_cond, atom_sequence_decoding_body, [i, i_len, cells, hs, computed_ens], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None, None])])
  return computed_ens
  
  
def compute_sequence_match(seq1, seq2):
  return tf.reduce_sum(tf.cast(tf.equal(seq1, seq2), int_type))
  

def compute_beam_accurates(computed_ens, oracle_atoms):
  
  def one_atom_sequence_cond(i, i_len, *_):
    return tf.less(i, i_len)
  
  def one_atom_sequence_body(i, i_len, accurates):
    computed_en = computed_ens[i]
    accurate = compute_sequence_match(computed_en, oracle_atoms)
    accurates = tf.concat([accurates, [accurate]], axis=0)
    return i + 1, i_len, accurates
  
  accurates = tf.zeros([0], int_type)
  _, _, accurates = tf.while_loop(one_atom_sequence_cond, one_atom_sequence_body, [0, tf.shape(computed_ens)[0], accurates], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])], parallel_iterations=1)
  
  o_len = tf.shape(oracle_atoms)[-1]
  exact_match = tf.reduce_sum(tf.cast(tf.equal(accurates, o_len), int_type))
  max_index = tf.argmax(accurates, axis=0)
  mrr = tf.constant(1.0, float_type) / (tf.cast(max_index, float_type) + 1.0) * tf.cast(exact_match, float_type)
  r_accurates = tf.cast(accurates, float_type) / tf.cast(o_len, float_type)
  accurate = tf.zeros([0], float_type)
  for btk in top_ks:
    r_slice_size = tf.minimum(btk, tf.shape(r_accurates)[-1])
    btk_accurate = tf.reduce_max(tf.slice(r_accurates, [0], [r_slice_size]), axis=0)
    accurate = tf.concat([accurate, [btk_accurate]], axis=0)
  return mrr, accurate






