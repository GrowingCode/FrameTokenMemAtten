import tensorflow as tf
from metas.non_hyper_constants import float_type, int_type,\
  all_skt_pe_to_each_end, all_skt_pe_to_each_start, all_skt_one_to_each_end,\
  all_skt_one_to_each_start
from metas.hyper_settings import num_units, top_ks, skeleton_as_one,\
  skeleton_decode_way, skeleton_as_pair_encoded, skeleton_as_each,\
  skeleton_seq_accuracy_mode, skeleton_seq_accuracy_based_on_each_atom,\
  skeleton_seq_accuracy_based_on_one_whole
from models.loss_accurate import compute_logits_given_to_deocde_embed_with_computed_embeddings
from utils.cartesian_util import cartesian_add_one_dim_vector,\
  cartesian_concat_two_dim_mats


def compute_beam_sequences(linear_atom_output_w, atom_lstm, atom_embedder, begin_cell, begin_h, atom_length):
     
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
      _, (n_cell, n_h) = atom_lstm(atom_embedder.compute_h(n_atom_ens[i]), (cell, h))
      n_cells = tf.concat([n_cells, n_cell], axis=0)
      n_hs = tf.concat([n_hs, n_h], axis=0)
      n_computed_en = tf.concat([computed_en, [n_atom_ens[i]]], axis=0)
      n_computed_ens = tf.concat([n_computed_ens, [n_computed_en]], axis=0)
    return n_cells, n_hs, n_probs, n_computed_ens
  
  def atom_sequence_decoding_cond(i, i_len, *_):
    return tf.less(i, i_len)
  
  def atom_sequence_decoding_body(i, i_len, cells, hs, probs, computed_ens):
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
  probs = tf.zeros([1], float_type)
  computed_ens = tf.zeros([1, 0], int_type)
  _, _, _, _, _, computed_ens = tf.while_loop(atom_sequence_decoding_cond, atom_sequence_decoding_body, [i, i_len, cells, hs, probs, computed_ens], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None]), tf.TensorShape([None, None])])
  return computed_ens


def compute_accuracy_of_sequences(type_content_data, computed_en_seqs, oracle_computed_en_seq):
  
  if skeleton_decode_way == skeleton_as_one:
    o_en_e_start = tf.gather(type_content_data[all_skt_one_to_each_start], [oracle_computed_en_seq])
    o_en_e_end = tf.gather(type_content_data[all_skt_one_to_each_end], [oracle_computed_en_seq])
    e_lens = o_en_e_end - o_en_e_start + 1
  elif skeleton_decode_way == skeleton_as_pair_encoded:
    o_en_e_start = tf.gather(type_content_data[all_skt_pe_to_each_start], [oracle_computed_en_seq])
    o_en_e_end = tf.gather(type_content_data[all_skt_pe_to_each_end], [oracle_computed_en_seq])
    e_lens = o_en_e_end - o_en_e_start + 1
  elif skeleton_decode_way == skeleton_as_each:
    e_lens = tf.ones([tf.shape(oracle_computed_en_seq)[-1]], int_type)
  
  e_lens = tf.cast(e_lens, float_type)
  eq_all_lens = tf.reduce_sum(e_lens)
  
  def compute_acc_cond(i, i_len, *_):
    return i < i_len
  
  def compute_acc_body(i, i_len, epos_acc, whole_acc):
    one_computed_en_seq = computed_en_seqs[i]
    tc = tf.zeros([tf.shape(oracle_computed_en_seq)[-1] - tf.shape(one_computed_en_seq)[-1]], int_type) - 1
    r_one_computed_en_seq = tf.concat([one_computed_en_seq, tc], axis=0)
    
    eq = tf.cast(tf.equal(r_one_computed_en_seq, oracle_computed_en_seq), float_type)
    eq_lens = tf.reduce_sum(eq * e_lens)
#     eq_all_acc = tf.reduce_sum(eq)
#     eq_all_count = tf.cast(tf.shape(eq)[-1], float_type)
    if skeleton_seq_accuracy_mode == skeleton_seq_accuracy_based_on_one_whole:
      epos_right = eq_lens / eq_all_lens
      whole_right = tf.cast(tf.equal(eq_lens, eq_all_lens), float_type)
    elif skeleton_seq_accuracy_mode == skeleton_seq_accuracy_based_on_each_atom:
      epos_right = eq_lens
      whole_right = tf.cast(tf.equal(eq_lens, eq_all_lens), float_type) * eq_all_lens
    else:
      assert False
    
    epos_r_acc = tf.maximum(epos_acc[-1], epos_right)
    whole_r_acc = tf.maximum(whole_acc[-1], whole_right)
    
    epos_acc = tf.concat([epos_acc, [epos_r_acc]], axis=0)
    whole_acc = tf.concat([whole_acc, [whole_r_acc]], axis=0)
    
    return i+1, i_len, epos_acc, whole_acc
  
  n = tf.shape(computed_en_seqs)[0]
  each_acc = tf.zeros([1], float_type)
  whole_acc = tf.zeros([1], float_type)
  _, _, each_acc, whole_acc = tf.while_loop(compute_acc_cond, compute_acc_body, [0, n, each_acc, whole_acc], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None])])
  
  f_each_acc = tf.zeros([0], float_type)
  f_whole_acc = tf.zeros([0], float_type)
  acc_len = tf.shape(each_acc)[-1]
  tpk_len = len(top_ks)
  for i in range(tpk_len):
    tpk = top_ks[i]
    r_sel = tf.minimum(tpk, acc_len-1)
    f_each_acc = tf.concat([f_each_acc, [each_acc[r_sel]]], axis=0)
    f_whole_acc = tf.concat([f_whole_acc, [whole_acc[r_sel]]], axis=0)
  
  if skeleton_seq_accuracy_mode == skeleton_seq_accuracy_based_on_one_whole:
    f_count = tf.constant(1, int_type)
  elif skeleton_seq_accuracy_mode == skeleton_seq_accuracy_based_on_each_atom:
    f_count = eq_all_lens
  else:
    assert False
  
  return f_each_acc, f_whole_acc, f_count


def dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens):
  
  def compute_ens_cond(i, i_len, *_):
    return i < i_len
  
  def compute_ens_body(i, i_len, acc_log_probs, acc_ens):
    o_prob = o_log_probs[i]
    o_en = o_ens[i]
    
    fa_log_probs = cartesian_add_one_dim_vector(acc_log_probs, o_prob)
    fa_ens = cartesian_concat_two_dim_mats(acc_ens, tf.expand_dims(o_en, axis=1))
    
    _, indices = tf.nn.top_k(fa_log_probs, top_ks[-1])
    sorted_probs = fa_log_probs[indices]
    sorted_ens = fa_ens[indices]
    
    return i+1, i_len, sorted_probs, sorted_ens 
  
  seq_len = tf.shape(o_log_probs)[0]
  acc_log_probs = tf.zeros([0, top_ks[-1]], float_type)
  acc_ens = tf.zeros([0, top_ks[-1]], int_type)
  _, _, acc_log_probs, acc_ens = tf.while_loop(compute_ens_cond, compute_ens_body, [tf.constant(0, int_type), seq_len, acc_log_probs, acc_ens], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, top_ks[-1]]), tf.TensorShape([None, top_ks[-1]])], parallel_iterations=1)
  return acc_ens


# def compute_accuracy_of_distinct_parallel_tokens(o_log_probs, o_ens, oracle_computed_en_seq):
#   computed_en_seqs = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
#   return compute_accuracy_of_sequences(computed_en_seqs, oracle_computed_en_seq)
  









