from metas.hyper_settings import use_dup_model, \
  token_memory_mode, decode_attention_way,\
  decode_no_attention, token_valid_mode,\
  token_in_scope_valid, token_meaningful_valid, decode_with_attention,\
  no_memory_mode, top_ks, concat_memory_mode, only_consider_var_accuracy,\
  consider_all_token_accuracy, only_consider_unseen_var_accuracy,\
  token_accuracy_mode, only_memory_mode, abs_size_concat_memory_mode,\
  abs_size_var_novar_all_concat_memory_mode, only_consider_non_var_accuracy,\
  only_consider_token_kind_accuracy, token_kind_consider_range_mode,\
  token_kind_default_range, token_kind_simplename_range,\
  token_kind_simplename_approximate_variable_range,\
  token_kind_non_leaf_at_least_two_children_without_qualified_node
from metas.non_hyper_constants import int_type, float_type, all_token_summary,\
  TokenHitNum, UNK_en, bool_type, default_token_kind,\
  simplename_approximate_not_variable, simplename_approximate_variable,\
  non_leaf_at_least_two_children_without_qualified_node
from models.loss_accurate import compute_loss_and_accurate_from_linear_with_computed_embeddings
import tensorflow as tf


# def decode_one_token(type_content_data, training, oracle_type_content_en, oracle_type_content_var, oracle_type_content_var_relative, metrics_index, token_metrics, linear_token_output_w, token_lstm, token_embedder, token_attention, dup_token_lstm=None, dup_token_embedder=None, token_pointer=None):
#   if token_valid_mode == token_in_scope_valid:
#     en_valid_bool = tf.less(oracle_type_content_en, type_content_data[all_token_summary][TokenHitNum])
#   elif token_valid_mode == token_meaningful_valid:
#     en_valid_bool = tf.logical_and(tf.greater(oracle_type_content_en, 2), tf.less(oracle_type_content_en, type_content_data[all_token_summary][TokenHitNum]))
#   else:
#     assert False
#   en_valid = tf.cast(en_valid_bool, float_type)
#   out_use_en = tf.stack([UNK_en, oracle_type_content_en])[tf.cast(en_valid_bool, int_type)]
#   ''' typical token swords prediction '''
#   cell = tf.expand_dims(token_metrics[metrics_index["token_accumulated_cell"]][-1], 0)
#   h = tf.expand_dims(token_metrics[metrics_index["token_accumulated_h"]][-1], 0)
#   ''' for attention use '''
# #   acc_ens = token_metrics[metrics_index["token_accumulated_en"]]
#   if decode_attention_way == decode_no_attention:
#     out_use_h = h
#   elif decode_attention_way == decode_stand_attention:
#     n_size = tf.shape(token_metrics[metrics_index["token_accumulated_h"]])[0]
#     acc_hs = tf.slice(token_metrics[metrics_index["token_accumulated_h"]], [0, 0], [n_size - 1, num_units])
#     out_use_h = token_attention.compute_attention_h(acc_hs, h)
#   elif decode_attention_way == decode_memory_attention:
#     out_use_h = token_attention.compute_attention_h(token_metrics[metrics_index["memory_hs"]], h)
#   elif decode_attention_way == decode_memory_concat_attention:
#     out_use_h = token_attention.compute_attention_h(token_metrics[metrics_index["memory_concat_h"]], h)
#   
#   before_accurate = token_metrics[metrics_index["all_accurate"]]
#   before_mrr = token_metrics[metrics_index["all_mrr"]]
#   before_token_accurate = token_metrics[metrics_index["token_accurate"]]
#   before_token_mrr = token_metrics[metrics_index["token_mrr"]]
#   ''' decode and compute sword level accurate '''
#   mrr_of_this_node, accurate_of_this_node, loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(training, linear_token_output_w, out_use_en, out_use_h)
#   token_metrics[metrics_index["token_loss"]] = token_metrics[metrics_index["token_loss"]] + loss_of_this_node * en_valid
#   token_metrics[metrics_index["token_accurate"]] = token_metrics[metrics_index["token_accurate"]] + accurate_of_this_node * en_valid
#   token_metrics[metrics_index["token_mrr"]] = token_metrics[metrics_index["token_mrr"]] + mrr_of_this_node * en_valid
#   token_metrics[metrics_index["token_lm_loss"]] = token_metrics[metrics_index["token_lm_loss"]] + loss_of_this_node * en_valid
#   token_metrics[metrics_index["token_lm_accurate"]] = token_metrics[metrics_index["token_lm_accurate"]] + accurate_of_this_node * en_valid
#   token_metrics[metrics_index["token_lm_mrr"]] = token_metrics[metrics_index["token_lm_mrr"]] + mrr_of_this_node * en_valid
#   token_metrics[metrics_index["token_count"]] = token_metrics[metrics_index["token_count"]] + 1
#   token_metrics[metrics_index["all_loss"]] = token_metrics[metrics_index["all_loss"]] + loss_of_this_node * en_valid
#   token_metrics[metrics_index["all_accurate"]] = token_metrics[metrics_index["all_accurate"]] + accurate_of_this_node * en_valid
#   token_metrics[metrics_index["all_mrr"]] = token_metrics[metrics_index["all_mrr"]] + mrr_of_this_node * en_valid
#   token_metrics[metrics_index["all_count"]] = token_metrics[metrics_index["all_count"]] + 1
#   
#   after_accurate = token_metrics[metrics_index["all_accurate"]]
#   after_mrr = token_metrics[metrics_index["all_mrr"]]
#   after_token_accurate = token_metrics[metrics_index["token_accurate"]]
#   after_token_mrr = token_metrics[metrics_index["token_mrr"]]
#   
#   ''' dup token prediction '''
#   if use_dup_model:
#     ''' compute duplicate pattern '''
#     dup_cell = tf.expand_dims(token_metrics[metrics_index["dup_token_accumulated_cell"]][-1], 0)
#     dup_h = tf.expand_dims(token_metrics[metrics_index["dup_token_accumulated_h"]][-1], 0)
#     dup_n_size = tf.shape(token_metrics[metrics_index["dup_token_accumulated_h"]])[0]
#     dup_acc_hs = tf.slice(token_metrics[metrics_index["dup_token_accumulated_h"]], [0, 0], [dup_n_size - 1, num_units])
#     dup_acc_ens = token_metrics[metrics_index["token_accumulated_en"]]
#     r_var_relative = oracle_type_content_var_relative
#     
#     if compute_token_memory:
#       if token_memory_mode == only_memory_mode:
#         dup_acc_hs = token_metrics[metrics_index["dup_memory_hs"]]
#         dup_acc_ens = token_metrics[metrics_index["memory_accumulated_en"]]
#         r_var_relative = tf.shape(dup_acc_hs)[0] - oracle_type_content_var
#       elif token_memory_mode == memory_concat_mode:
#         ''' other parameters inherit from the standard duplicate pattern '''
# #         dup_m_size = tf.shape(token_metrics[metrics_index["dup_memory_concat_h"]])[0]
# #         a_op = tf.Assert(tf.equal(dup_n_size-1, dup_m_size), ["Strange error, two accumulated_h should be equal!"])
# #         p_op = tf.print(["dup_n_size-1:", dup_n_size-1, "dup_m_size:", dup_m_size])
# #         with tf.control_dependencies([a_op, p_op]):
#         dup_acc_hs = token_metrics[metrics_index["dup_memory_concat_h"]]
# #         dup_acc_hs = tf.slice(token_metrics[metrics_index["dup_memory_concat_h"]], [0, 0], [dup_n_size - 1, num_units])
#       else:
#         assert False, "error! unrecognized token_memory_mode!"
#       
# #     r_var_relative_valid = tf.cast(tf.greater(r_var_relative, 0), int_type)
# #     a_r_op = tf.Assert(tf.logical_or(tf.cast(1-r_var_relative_valid, bool_type), tf.equal(dup_acc_ens[tf.shape(dup_acc_ens)[0]-r_var_relative*r_var_relative_valid-1+r_var_relative_valid], oracle_type_content_en)), ["invalid variable relative"])
# #     a_op = tf.assert_equal(tf.shape(dup_acc_ens)[0], tf.shape(dup_acc_hs)[0])
# #     p_op = tf.print(["dup_acc_ens_size:", tf.shape(dup_acc_ens)[0], "dup_acc_hs_size:", tf.shape(dup_acc_hs)[0], "dup_acc_ens:", dup_acc_ens, "oracle_type_content_en:", oracle_type_content_en, "oracle_type_content_var:", oracle_type_content_var, "r_var_relative:", r_var_relative], summarize = 500)
# #     with tf.control_dependencies([p_op]):
#     dup_logits, neg_dup_logits, neg_ele_logit, dup_max_arg_acc_h, dup_min_cared_h = token_pointer.compute_logits(dup_acc_hs, dup_h)
#     is_dup_logits = token_pointer.compute_is_dup_logits(dup_max_arg_acc_h, dup_min_cared_h, dup_h)
#     dup_mrr_of_this_node, dup_accurate_of_this_node, dup_loss_of_this_node, r_dup_mrr, r_dup_accurate, predict_to_use_pre_exist = token_pointer.compute_dup_loss(training, dup_acc_ens, oracle_type_content_en, r_var_relative, is_dup_logits, dup_logits, neg_dup_logits, neg_ele_logit)
#     
#     token_metrics[metrics_index["token_dup_loss"]] = token_metrics[metrics_index["token_dup_loss"]] + dup_loss_of_this_node
#     token_metrics[metrics_index["token_dup_accurate"]] = token_metrics[metrics_index["token_dup_accurate"]] + dup_accurate_of_this_node
#     token_metrics[metrics_index["token_dup_mrr"]] = token_metrics[metrics_index["token_dup_mrr"]] + dup_mrr_of_this_node
#     token_metrics[metrics_index["all_loss"]] = token_metrics[metrics_index["all_loss"]] + dup_loss_of_this_node
#     ''' compute accurate '''
#     token_metrics[metrics_index["all_accurate"]] = before_accurate + tf.stack([after_accurate - before_accurate, r_dup_accurate], axis=0)[predict_to_use_pre_exist]
#     token_metrics[metrics_index["all_mrr"]] = before_mrr + tf.stack([after_mrr - before_mrr, r_dup_mrr], axis=0)[predict_to_use_pre_exist]
#     token_metrics[metrics_index["token_accurate"]] = before_token_accurate + tf.stack([after_token_accurate - before_token_accurate, r_dup_accurate], axis=0)[predict_to_use_pre_exist]
#     token_metrics[metrics_index["token_mrr"]] = before_token_mrr + tf.stack([after_token_mrr - before_token_mrr, r_dup_mrr], axis=0)[predict_to_use_pre_exist]
#     
#     _, (new_dup_token_cell, new_dup_token_h) = dup_token_lstm(dup_token_embedder.compute_h(oracle_type_content_en), (dup_cell, dup_h))
#     token_metrics[metrics_index["dup_token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["dup_token_accumulated_cell"]], new_dup_token_cell, accumulated_token_max_length)
#     token_metrics[metrics_index["dup_token_accumulated_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["dup_token_accumulated_h"]], new_dup_token_h, accumulated_token_max_length)
#     
#     if compute_token_memory:
#       if token_memory_mode == memory_concat_mode:
#         token_metrics[metrics_index["dup_memory_concat_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["dup_memory_concat_cell"]], [token_metrics[metrics_index["dup_memory_concat_cell"]][0]], accumulated_token_max_length)
#         token_metrics[metrics_index["dup_memory_concat_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["dup_memory_concat_h"]], [token_metrics[metrics_index["dup_memory_concat_h"]][0]], accumulated_token_max_length)
#     
#   token_metrics[metrics_index["token_accumulated_en"]] = concat_in_fixed_length_one_dimension(token_metrics[metrics_index["token_accumulated_en"]], [oracle_type_content_en], accumulated_token_max_length)
#   ''' predict next node '''
#   _, (new_cell, new_h) = token_lstm(token_embedder.compute_h(oracle_type_content_en), (cell, h))
#   token_metrics[metrics_index["token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_cell"]], new_cell, accumulated_token_max_length)
#   token_metrics[metrics_index["token_accumulated_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_h"]], new_h, accumulated_token_max_length)
#   return tuple(token_metrics)
# 
# 
# def decode_swords_of_one_token(type_content_data, training, token_en, token_atom_sequence, metrics_index, metrics_shape, token_metrics, token_lstm, token_embedder, linear_sword_output_w, sword_embedder, sword_lstm):
# #   atom_length_in_float = tf.cast(tf.shape(token_atom_sequence)[-1], float_type)
#   assert type_content_data != None
#   
#   def decode_one_sword_cond(w, w_len, *_):
#     return tf.less(w, w_len)
#    
#   def decode_one_sword_body(w, w_len, sword_c, sword_h, *sword_metrics_tuple):
#     sword_metrics = list(sword_metrics_tuple)
#     oracle_sword_en = token_atom_sequence[w]
#     mrr_of_this_node, accurate_of_this_node, loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(training, linear_sword_output_w, oracle_sword_en, sword_h)
#     ''' predict next node '''
#     e_embed = sword_embedder.compute_h(oracle_sword_en)
#     _, (new_sword_c, new_sword_h) = sword_lstm(e_embed, (sword_c, sword_h))
#     ''' final loss and accurate '''
#     sword_metrics[metrics_index["sword_loss"]] = sword_metrics[metrics_index["sword_loss"]] + loss_of_this_node  # + dup_loss_of_this_node
#     sword_metrics[metrics_index["sword_accurate"]] = sword_metrics[metrics_index["sword_accurate"]] + accurate_of_this_node
#     sword_metrics[metrics_index["sword_mrr"]] = sword_metrics[metrics_index["sword_mrr"]] + mrr_of_this_node
#     sword_metrics[metrics_index["sword_count"]] = sword_metrics[metrics_index["sword_count"]] + 1
#     
#     sword_metrics[metrics_index["all_loss"]] = sword_metrics[metrics_index["all_loss"]] + loss_of_this_node# / atom_length_in_float
#     sword_metrics[metrics_index["all_accurate"]] = sword_metrics[metrics_index["all_accurate"]] + accurate_of_this_node# / atom_length_in_float
#     sword_metrics[metrics_index["all_mrr"]] = sword_metrics[metrics_index["all_mrr"]] + mrr_of_this_node# / atom_length_in_float
#     sword_metrics[metrics_index["all_count"]] = sword_metrics[metrics_index["all_count"]] + 1
#     return (w + 1, w_len, new_sword_c, new_sword_h, *sword_metrics)
#   
#   cell = tf.expand_dims(token_metrics[metrics_index["token_accumulated_cell"]][-1], 0)
#   h = tf.expand_dims(token_metrics[metrics_index["token_accumulated_h"]][-1], 0)
#   
# #   if training:
# #     beam_mrr_of_this_node, beam_accurate_of_this_node = tf.constant(0.0, float_type), tf.zeros([len(top_ks)], float_type)
# #   else:
# #     beam_mrr_of_this_node, beam_accurate_of_this_node = compute_atom_sequence_beam_accurate(linear_sword_output_w, sword_lstm, sword_embedder, cell, h, token_atom_sequence)
#   t_array = token_metrics[metrics_index["atom_beam"]]
#   token_metrics[metrics_index["atom_beam"]] = t_array.write(t_array.size(), compute_swords_prediction(linear_sword_output_w, sword_lstm, sword_embedder, cell, h, token_atom_sequence))
#   
#   _, (new_cell, new_h) = token_lstm(token_embedder.compute_h(token_en), (cell, h))
#   token_metrics[metrics_index["token_accumulated_cell"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_cell"]], new_cell, accumulated_token_max_length)
#   token_metrics[metrics_index["token_accumulated_h"]] = concat_in_fixed_length_two_dimension(token_metrics[metrics_index["token_accumulated_h"]], new_h, accumulated_token_max_length)
#   
#   token_metrics_res = tf.while_loop(decode_one_sword_cond, decode_one_sword_body, [0, tf.shape(token_atom_sequence)[-1], cell, h, *token_metrics], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([1, num_units]), *metrics_shape], parallel_iterations=1)
#   r_token_metrics = list(token_metrics_res[4:])
#   r_token_metrics[metrics_index["token_count"]] = r_token_metrics[metrics_index["token_count"]] + 1
#   return r_token_metrics
# 
# 
# def compute_swords_prediction(linear_sword_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, oracle_atoms):
#   atom_length = tf.shape(oracle_atoms)[-1]
#   computed_ens = compute_beam_sequences(linear_sword_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, atom_length)
#   return computed_ens
# 
# 
# # def compute_atom_sequence_beam_accurate(linear_sword_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, oracle_atoms):
# #   atom_length = tf.shape(oracle_atoms)[-1]
# #   computed_ens = compute_beam_sequences(linear_sword_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, atom_length)
# #   mrr, accurate = compute_beam_accurates(computed_ens, oracle_atoms)
# #   return mrr, accurate
# 
# 
# def compute_beam_sequences(linear_atom_output_w, sword_lstm, sword_embedder, begin_cell, begin_h, atom_length):
#     
#   def atom_decode_one(top_n, cell, h, prob_scalar, computed_en): 
#     n_cells = tf.zeros([0, num_units], float_type)
#     n_hs = tf.zeros([0, num_units], float_type)
#     logits = compute_logits_given_to_deocde_embed_with_computed_embeddings(linear_atom_output_w, h)
#     probs = tf.nn.softmax(logits)
#     probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
#     _, n_atom_ens = tf.nn.top_k(probs, top_n)
#     n_probs = tf.gather(probs, n_atom_ens) + prob_scalar
#     n_computed_ens = tf.zeros([0, tf.shape(computed_en)[-1] + 1], int_type)
#     for _ in range(top_n):
#       _, (n_cell, n_h) = sword_lstm(sword_embedder.compute_h(n_atom_ens[i]), (cell, h))
#       n_cells = tf.concat([n_cells, n_cell], axis=0)
#       n_hs = tf.concat([n_hs, n_h], axis=0)
#       n_computed_en = tf.concat([computed_en, [n_atom_ens[i]]], axis=0)
#       n_computed_ens = tf.concat([n_computed_ens, [n_computed_en]], axis=0)
#     return n_cells, n_hs, n_probs, n_computed_ens  # n_accurates, 
#   
#   def atom_sequence_decoding_cond(i, i_len, *_):
#     return tf.less(i, i_len)
# 
#   ''' body '''
# 
#   def atom_sequence_decoding_body(i, i_len, cells, hs, probs, computed_ens):
#     beam_size = top_ks[-1]
#     
#     ''' use h to decode '''
#     ''' next char embedding '''
#     
#     def atom_decode_filter(cells, hs, probs, atom_ens, retain_size):
#       r_size = tf.minimum(tf.shape(probs)[0], retain_size)
#       _, indices = tf.nn.top_k(probs, r_size)
#       return tf.gather(cells, indices), tf.gather(hs, indices), tf.gather(probs, indices), tf.gather(atom_ens, indices)  # , tf.gather(accurates, indices)
#   
#     def beam_cond(j, j_len, *_):
#       return tf.less(j, j_len)
#     
#     def beam_body(j, j_len, candi_cells, candi_hs, candi_atom_probs, candi_atom_ens):  # candi_accurates, 
#       n_cells, n_hs, n_probs, n_computed_ens = atom_decode_one(beam_size, [cells[j]], [hs[j]], probs[j], computed_ens[j])  # , computed_ens[j][-1], n_accurates, accurates[j], oracle_atoms[i]
#       candi_cells = tf.concat([candi_cells, n_cells], axis=0)
#       candi_hs = tf.concat([candi_hs, n_hs], axis=0)
#       candi_atom_probs = tf.concat([candi_atom_probs, n_probs], axis=0)
#       candi_atom_ens = tf.concat([candi_atom_ens, n_computed_ens], axis=0)
#       return j + 1, j_len, candi_cells, candi_hs, candi_atom_probs, candi_atom_ens  # candi_accurates, 
#     
#     j = tf.constant(0, int_type)
#     j_len = tf.shape(cells)[0]
#     candi_cells = tf.zeros([0, num_units], float_type)
#     candi_hs = tf.zeros([0, num_units], float_type)
#     candi_probs = tf.zeros([0], float_type)
#     candi_atom_ens = tf.zeros([0, tf.shape(computed_ens)[-1] + 1], int_type)
#     _, _, candi_cells, candi_hs, candi_probs, candi_atom_ens = tf.while_loop(beam_cond, beam_body, [j, j_len, candi_cells, candi_hs, candi_probs, candi_atom_ens], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None]), tf.TensorShape([None, None])], parallel_iterations=1)
#     candi_cells, candi_hs, candi_probs, candi_atom_ens = atom_decode_filter(candi_cells, candi_hs, candi_probs, candi_atom_ens, beam_size)
#     return i + 1, i_len, candi_cells, candi_hs, candi_probs, candi_atom_ens  # , candi_accurates
#   
#   ''' compute loss '''
#   i = tf.constant(0, int_type)
#   i_len = atom_length  # tf.shape(oracle_atoms)[-1]
#   cells = begin_cell
#   hs = begin_h
#   probs = tf.zeros([1], float_type)
#   computed_ens = tf.zeros([1, 0], int_type)
#   _, _, _, _, _, computed_ens = tf.while_loop(atom_sequence_decoding_cond, atom_sequence_decoding_body, [i, i_len, cells, hs, probs, computed_ens], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None]), tf.TensorShape([None, None])])
#   return computed_ens
#   
#   
# def compute_sequence_match(seq1, seq2):
#   return tf.reduce_sum(tf.cast(tf.equal(seq1, seq2), int_type))
#   
# 
# def compute_beam_accurates(computed_ens, oracle_atoms):
#   
#   def one_atom_sequence_cond(i, i_len, *_):
#     return tf.less(i, i_len)
#   
#   def one_atom_sequence_body(i, i_len, accurates):
#     computed_en = computed_ens[i]
#     accurate = compute_sequence_match(computed_en, oracle_atoms)
#     accurates = tf.concat([accurates, [accurate]], axis=0)
#     return i + 1, i_len, accurates
#   
#   accurates = tf.zeros([0], int_type)
#   _, _, accurates = tf.while_loop(one_atom_sequence_cond, one_atom_sequence_body, [0, tf.shape(computed_ens)[0], accurates], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])], parallel_iterations=1)
#   
#   o_len = tf.shape(oracle_atoms)[-1]
#   exact_match = tf.reduce_sum(tf.cast(tf.equal(accurates, o_len), int_type))
#   max_index = tf.argmax(accurates, axis=0)
#   mrr = tf.constant(1.0, float_type) / (tf.cast(max_index, float_type) + 1.0) * tf.cast(exact_match, float_type)
#   r_accurates = tf.cast(accurates, float_type) / tf.cast(o_len, float_type)
#   accurate = tf.zeros([0], float_type)
#   for btk in top_ks:
#     r_slice_size = tf.minimum(btk, tf.shape(r_accurates)[-1])
#     btk_accurate = tf.reduce_max(tf.slice(r_accurates, [0], [r_slice_size]), axis=0)
#     accurate = tf.concat([accurate, [btk_accurate]], axis=0)
#   return mrr, accurate

class TokenDecoder():
  
  def __init__(self, type_content_data, metrics_index, linear_token_output_w, token_attention, dup_token_pointer=None):
    self.type_content_data = type_content_data
    self.metrics_index = metrics_index
    self.linear_token_output_w = linear_token_output_w
    self.token_attention = token_attention
    self.dup_token_pointer = dup_token_pointer
    
  def decode_one_token(self, token_metrics, training, oracle_type_content_en, oracle_type_content_var, oracle_type_content_var_relative, oracle_type_content_kind):
    if token_accuracy_mode == consider_all_token_accuracy:
      t_valid_bool = tf.constant(True, bool_type)
    elif token_accuracy_mode == only_consider_token_kind_accuracy:
      t_valid_bool = is_in_token_kind_range(oracle_type_content_kind)
    elif token_accuracy_mode == only_consider_var_accuracy:
      t_valid_bool = tf.greater(oracle_type_content_var, 0)
    elif token_accuracy_mode == only_consider_unseen_var_accuracy:
      t_valid_bool = tf.logical_and(oracle_type_content_var > 0, tf.greater_equal(oracle_type_content_en, self.type_content_data[all_token_summary][TokenHitNum]))
    elif token_accuracy_mode == only_consider_non_var_accuracy:
      t_valid_bool = tf.less_equal(oracle_type_content_var, 0)
    else:
      assert False
    t_valid = tf.cast(t_valid_bool, float_type)
    t_valid_int = tf.cast(t_valid_bool, int_type)
    
    if token_valid_mode == token_in_scope_valid:
      en_valid_bool = tf.less(oracle_type_content_en, self.type_content_data[all_token_summary][TokenHitNum])
    elif token_valid_mode == token_meaningful_valid:
      en_valid_bool = tf.logical_and(tf.greater(oracle_type_content_en, 2), tf.less(oracle_type_content_en, self.type_content_data[all_token_summary][TokenHitNum]))
    else:
      assert False
    en_valid = tf.cast(en_valid_bool, float_type)
    out_use_en = tf.stack([UNK_en, oracle_type_content_en])[tf.cast(en_valid_bool, int_type)]
    
    ''' typical token swords prediction '''
    h = token_metrics[self.metrics_index["token_h"]]
    if decode_attention_way == decode_no_attention:
      out_use_h = h
    elif decode_attention_way == decode_with_attention:
      assert token_memory_mode > no_memory_mode
      out_use_h = self.token_attention.compute_attention_h(token_metrics[self.metrics_index["memory_acc_h"]], h)
    else:
      assert False
    mrr_of_this_node, accurate_of_this_node, loss_of_this_node = compute_loss_and_accurate_from_linear_with_computed_embeddings(training, self.linear_token_output_w, out_use_en, out_use_h)
    loss_of_this_node = loss_of_this_node * en_valid
    accurate_of_this_node = accurate_of_this_node * en_valid * t_valid
    mrr_of_this_node = mrr_of_this_node * en_valid * t_valid
    
    token_metrics[self.metrics_index["token_lm_loss"]] += loss_of_this_node
    token_metrics[self.metrics_index["token_lm_accurate"]] += accurate_of_this_node
    token_metrics[self.metrics_index["token_lm_mrr"]] += mrr_of_this_node
    token_metrics[self.metrics_index["all_loss"]] += loss_of_this_node
    
    dup_repeat_mrr_of_this_node = tf.constant(0.0, float_type)
    dup_repeat_accurate_of_this_node = tf.zeros([len(top_ks)], float_type)
    predict_to_use_pre_exist = 0
    
    if use_dup_model:
      assert token_memory_mode > no_memory_mode
      dup_h = token_metrics[self.metrics_index["dup_token_h"]]
      dup_acc_hs = token_metrics[self.metrics_index["dup_memory_acc_h"]]
      dup_acc_ens = token_metrics[self.metrics_index["dup_memory_en"]]
      if token_memory_mode == only_memory_mode:
        r_var_relative = tf.shape(dup_acc_hs)[0] - oracle_type_content_var
      else:
        r_var_relative = oracle_type_content_var_relative
        assert token_memory_mode == concat_memory_mode or token_memory_mode == abs_size_concat_memory_mode or token_memory_mode == abs_size_var_novar_all_concat_memory_mode
      
#       p_op = tf.print(["dup_acc_ens:", dup_acc_ens, "var:", oracle_type_content_var, "r_var_relative:", r_var_relative], summarize=100)
#       with tf.control_dependencies([p_op]):
      dup_logits, neg_dup_logits, neg_ele_logit, dup_max_arg_acc_h, dup_min_cared_h = self.dup_token_pointer.compute_logits(dup_acc_hs, dup_h)
      is_dup_logits = self.dup_token_pointer.compute_is_dup_logits(dup_max_arg_acc_h, dup_min_cared_h, dup_h)
      dup_mrr_of_this_node, dup_accurate_of_this_node, dup_loss_of_this_node, dup_repeat_mrr_of_this_node, dup_repeat_accurate_of_this_node, predict_to_use_pre_exist = self.dup_token_pointer.compute_dup_loss(training, dup_acc_ens, oracle_type_content_en, r_var_relative, oracle_type_content_kind, is_dup_logits, dup_logits, neg_dup_logits, neg_ele_logit)
      dup_mrr_of_this_node = dup_mrr_of_this_node * t_valid
      dup_accurate_of_this_node = dup_accurate_of_this_node * t_valid
      predict_to_use_pre_exist = predict_to_use_pre_exist * t_valid_int
      
      token_metrics[self.metrics_index["token_dup_loss"]] += dup_loss_of_this_node
      token_metrics[self.metrics_index["token_dup_accurate"]] += dup_accurate_of_this_node
      token_metrics[self.metrics_index["token_dup_mrr"]] += dup_mrr_of_this_node
      token_metrics[self.metrics_index["all_loss"]] += dup_loss_of_this_node
      
    to_add_accurate_candidates = tf.stack([accurate_of_this_node, dup_repeat_accurate_of_this_node])
    token_metrics[self.metrics_index["all_accurate"]] += to_add_accurate_candidates[predict_to_use_pre_exist]
    to_add_mrr_candidates = tf.stack([mrr_of_this_node, dup_repeat_mrr_of_this_node])
    token_metrics[self.metrics_index["all_mrr"]] += to_add_mrr_candidates[predict_to_use_pre_exist]
    
    token_metrics[self.metrics_index["token_count"]] += 1 * t_valid_int
    token_metrics[self.metrics_index["all_count"]] += 1 * t_valid_int
    
    return token_metrics
  


def is_in_token_kind_range(oracle_en_kind):
  if token_kind_consider_range_mode == token_kind_default_range:
    ntc_bool = tf.equal(oracle_en_kind, default_token_kind)
  elif token_kind_consider_range_mode == token_kind_simplename_range:
    ntc_bool = tf.logical_or(tf.equal(oracle_en_kind, simplename_approximate_variable), tf.equal(oracle_en_kind, simplename_approximate_not_variable))
  elif token_kind_consider_range_mode == token_kind_simplename_approximate_variable_range:
    ntc_bool = tf.equal(oracle_en_kind, simplename_approximate_variable)
  elif token_kind_consider_range_mode == token_kind_non_leaf_at_least_two_children_without_qualified_node:
    ntc_bool = tf.equal(oracle_en_kind, non_leaf_at_least_two_children_without_qualified_node)
  else:
    assert False
  return ntc_bool



