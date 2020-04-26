import tensorflow as tf
from metas.non_hyper_constants import all_token_each_subword_sequence_start,\
  all_token_each_subword_sequence_end, all_token_subword_sequences, float_type,\
  all_token_summary, SwordHitNum, int_type, TokenHitNum, SkeletonHitNum, UNK_en,\
  learning_scope
from metas.hyper_settings import num_units, take_unseen_as_UNK
from models.embed_merger import EmbedMerger
from models.lstm import YLSTMCell
from utils.initializer import random_uniform_variable_initializer


def sword_sequence_for_token(type_content_data, oracle_token_en):
  atom_seq_start = type_content_data[all_token_each_subword_sequence_start][oracle_token_en]
  atom_seq_end = type_content_data[all_token_each_subword_sequence_end][oracle_token_en]
  return tf.slice(type_content_data[all_token_subword_sequences], [atom_seq_start], [atom_seq_end - atom_seq_start + 1])


class AtomEmbed():
  
  def __init__(self, type_content_data, vocab_embeddings):
    self.type_content_data = type_content_data
    self.vocab_embeddings = vocab_embeddings
  
  def compute_h_util(self, token_en, out_vocab_threshold):
    if take_unseen_as_UNK:
      out_of_vocab = tf.cast(token_en >= out_vocab_threshold, int_type)
      t_en = tf.stack([token_en, UNK_en])[out_of_vocab]
      res = self.vocab_embeddings[t_en]
    else:
      res = self.vocab_embeddings[token_en]
    res = tf.expand_dims(res, axis=0)
    return res


class SkeletonAtomEmbed(AtomEmbed):
  
  def __init__(self, type_content_data, vocab_embeddings):
    super(SkeletonAtomEmbed, self).__init__(type_content_data, vocab_embeddings)
  
  def compute_h(self, token_en):
    return self.compute_h_util(token_en, self.type_content_data[all_token_summary][SkeletonHitNum])


class TokenAtomEmbed(AtomEmbed):
  
  def __init__(self, type_content_data, vocab_embeddings):
    super(TokenAtomEmbed, self).__init__(type_content_data, vocab_embeddings)
  
  def compute_h(self, token_en):
    return self.compute_h_util(token_en, self.type_content_data[all_token_summary][TokenHitNum])


class SwordAtomEmbed(AtomEmbed):
  
  def __init__(self, type_content_data, vocab_embeddings):
    super(SwordAtomEmbed, self).__init__(type_content_data, vocab_embeddings)
  
  def compute_h(self, token_en):
    return self.compute_h_util(token_en, self.type_content_data[all_token_summary][SwordHitNum])


class BiLSTMEmbed():
  
  def __init__(self, type_content_data, vocab_embeddings):
    self.type_content_data = type_content_data
    self.vocab_embeddings = vocab_embeddings
    self.sword_embed = SwordAtomEmbed(self.type_content_data, self.vocab_embeddings)
    self.merger = EmbedMerger(101)
    self.forward_lstm = YLSTMCell(15)
    self.backward_lstm = YLSTMCell(25)
    with tf.variable_scope(learning_scope):
      self.ini_forward_cell = tf.get_variable("bi_embed_ini_forward_cell", shape=[1, num_units], dtype=float_type, initializer=random_uniform_variable_initializer(881, 882))
      self.ini_forward_h = tf.get_variable("bi_embed_ini_forward_h", shape=[1, num_units], dtype=float_type, initializer=random_uniform_variable_initializer(883, 884))
      self.ini_backward_cell = tf.get_variable("bi_embed_ini_backward_cell", shape=[1, num_units], dtype=float_type, initializer=random_uniform_variable_initializer(885, 886))
      self.ini_backward_h = tf.get_variable("bi_embed_ini_backward_h", shape=[1, num_units], dtype=float_type, initializer=random_uniform_variable_initializer(887, 888))
  
  def forward_backward_cell_h(self, look_up_indexes):
    
    def ea_forward_cond(e, e_len, *_):
      return tf.less(e, e_len)
     
    def ea_forward_body(e, e_len, cell, h, accumulated_cell, accumulated_h):
      e_embed = [self.sword_embed.compute_h(look_up_indexes[e])]
#       e_embed = [self.vocab_embeddings[look_up_indexes[e]]]
      new_cell, new_h = self.forward_lstm(e_embed, cell, h)
      accumulated_cell = tf.concat([accumulated_cell, cell], axis=0)
      accumulated_h = tf.concat([accumulated_h, h], axis=0)
      return e + 1, e_len, new_cell, new_h, accumulated_cell, accumulated_h
    
    def ea_backward_cond(e, e_start, *_):
      return tf.greater(e, e_start)
     
    def ea_backward_body(e, e_start, cell, h, accumulated_cell, accumulated_h):
      e_embed = [self.sword_embed.compute_h(look_up_indexes[e])]
#       e_embed = [self.vocab_embeddings[look_up_indexes[e]]]
      new_cell, new_h = self.backward_lstm(e_embed, cell, h)
      accumulated_cell = tf.concat([cell, accumulated_cell], axis=0)
      accumulated_h = tf.concat([h, accumulated_h], axis=0)
      return e - 1, e_start, new_cell, new_h, accumulated_cell, accumulated_h

    accumulated_forward_cell = tf.zeros([0, num_units], float_type)
    accumulated_forward_h = tf.zeros([0, num_units], float_type)
    _, _, forward_cell, forward_h, accumulated_forward_cell, accumulated_forward_h = tf.while_loop(ea_forward_cond, ea_forward_body, [0, tf.shape(look_up_indexes)[-1], self.ini_forward_cell, self.ini_forward_h, accumulated_forward_cell, accumulated_forward_h], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([1, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units])])
    accumulated_forward_cell = tf.concat([accumulated_forward_cell, forward_cell], axis=0)
    accumulated_forward_h = tf.concat([accumulated_forward_h, forward_h], axis=0)
    
    accumulated_backward_cell = tf.zeros([0, num_units], float_type)
    accumulated_backward_h = tf.zeros([0, num_units], float_type)
    _, _, backward_cell, backward_h, accumulated_backward_cell, accumulated_backward_h = tf.while_loop(ea_backward_cond, ea_backward_body, [tf.shape(look_up_indexes)[-1] - 1, -1, self.ini_backward_cell, self.ini_backward_h, accumulated_backward_cell, accumulated_backward_h], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([1, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units])])
    accumulated_backward_cell = tf.concat([backward_cell, accumulated_backward_cell], axis=0)
    accumulated_backward_h = tf.concat([backward_h, accumulated_backward_h], axis=0)
    
    return accumulated_forward_cell, accumulated_forward_h, accumulated_backward_cell, accumulated_backward_h
  
  def compute_h(self, token_en):
    loop_up_indexes = sword_sequence_for_token(self.type_content_data, token_en)
    _, accumulated_forward_h, _, accumulated_backward_h = self.forward_backward_cell_h(loop_up_indexes)
    c_forward_h = accumulated_forward_h[-1]
    c_backward_h = accumulated_backward_h[0]
    zero_mg = self.merger([c_forward_h], [c_backward_h])
    return zero_mg



