import tensorflow as tf
from metas.non_hyper_constants import float_type, int_type
from metas.hyper_settings import num_units, top_ks
from models.loss_accurate import compute_logits_given_to_deocde_embed_with_computed_embeddings


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


def compute_accuracy_of_sequences(computed_en_seqs, oracle_computed_en_seq):
  
  def compute_acc_cond(i, i_len, *_):
    return i < i_len
  
  def compute_acc_body(i, i_len, each_acc, whole_acc):
    one_computed_en_seq = computed_en_seqs[i]
    eq = tf.cast(tf.equal(one_computed_en_seq, oracle_computed_en_seq), float_type)
    
    todo
    return i+1, i_len, each_acc, whole_acc
  
  n = tf.shape(computed_en_seqs)[0]
  each_acc = tf.zeros([len(top_ks)], float_type);
  whole_acc = tf.zeros([len(top_ks)], float_type);
  _, _, each_acc, whole_acc = tf.while_loop(compute_acc_cond, compute_acc_body, [0, n, each_acc, whole_acc])
  return each_acc, whole_acc








