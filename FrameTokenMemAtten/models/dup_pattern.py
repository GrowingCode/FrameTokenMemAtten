import tensorflow as tf
from metas.non_hyper_constants import float_type, int_type
from metas.hyper_settings import top_ks, mrr_max, num_units, is_dup_mode,\
  simple_is_dup, mlp_is_dup, sigmoid_is_dup, attention_repetition_mode,\
  repetition_mode, max_repetition_mode, en_match, repetition_accuracy_mode,\
  exact_accurate, dup_share_parameters, dup_use_two_poles
from models.attention import YAttention
from utils.initializer import random_uniform_variable_initializer
from metas.tensor_constants import zero_tensor


class PointerNetwork():
  
  def __init__(self):
    if is_dup_mode == simple_is_dup:
      self.is_dup_w = tf.Variable(random_uniform_variable_initializer(200, 1080, [num_units, num_units]))
      self.is_not_dup_w = tf.Variable(random_uniform_variable_initializer(205, 1080, [num_units, num_units]))
    elif is_dup_mode == mlp_is_dup:
      self.is_dup_w = tf.Variable(random_uniform_variable_initializer(200, 1080, [2*num_units, 2*num_units]))
      self.is_dup_h = tf.Variable(random_uniform_variable_initializer(200, 1050, [1, 2*num_units]))
      self.is_not_dup_w = tf.Variable(random_uniform_variable_initializer(205, 1080, [2*num_units, 2*num_units]))
      self.is_not_dup_h = tf.Variable(random_uniform_variable_initializer(205, 1060, [1, 2*num_units]))
    elif is_dup_mode == sigmoid_is_dup:
      self.is_dup_h = tf.Variable(random_uniform_variable_initializer(200, 1050, [1, 2*num_units]))
      if dup_use_two_poles:
        self.two_poles_merge_w = tf.Variable(random_uniform_variable_initializer(200, 1060, [2*num_units, 1*num_units]))
        self.two_poles_merge_b = tf.Variable(random_uniform_variable_initializer(200, 1070, [1, 1*num_units]))
    else:
      assert False, "Unrecognized is_dup_mode!"
    self.dup_point_atten = YAttention(100)
    if repetition_mode == attention_repetition_mode:
      if dup_share_parameters:
        self.is_dup_point_atten = self.dup_point_atten
      else:
        self.is_dup_point_atten = YAttention(200)
    if dup_use_two_poles:
      self.neg_element = tf.Variable(random_uniform_variable_initializer(200, 2080, [1, 1*num_units]))
  
  def compute_logits(self, accumulated_h, h):
    dup_logits = self.dup_point_atten.compute_attention_logits(accumulated_h, h)
    neg_ele_logit = None
    if dup_use_two_poles:
      neg_ele_logit = self.dup_point_atten.compute_attention_logits(self.neg_element, h)
    if repetition_mode == max_repetition_mode:
      r_accumulated_h = accumulated_h
      r_dup_logits = dup_logits
      if dup_use_two_poles:
        r_accumulated_h = tf.concat([r_accumulated_h, self.neg_element], axis=0)
        r_dup_logits = tf.concat([dup_logits, neg_ele_logit], axis=0)
      dup_max_arg = tf.argmax(r_dup_logits)
      dup_max_arg = tf.cast(dup_max_arg, int_type)
      dup_max_cared_h = tf.expand_dims(r_accumulated_h[dup_max_arg], axis=0)
      dup_min_arg = tf.argmin(r_dup_logits)
      dup_min_arg = tf.cast(dup_min_arg, int_type)
      dup_min_cared_h = tf.expand_dims(r_accumulated_h[dup_min_arg], axis=0)
    elif repetition_mode == attention_repetition_mode:
      dup_max_cared_h = self.is_dup_point_atten.compute_attention_context(accumulated_h, h)
      dup_min_cared_h = zero_tensor
    else:
      assert False, "Unrecognized repetition_mode!"
    return dup_logits, neg_ele_logit, dup_max_cared_h, dup_min_cared_h
  
  def compute_is_dup_logits(self, dup_max_arg_acc_h, dup_min_arg_acc_h, h):
    if is_dup_mode == simple_is_dup:
      is_dup_logit = tf.squeeze(tf.matmul(tf.matmul(h, self.is_dup_w), dup_max_arg_acc_h, transpose_b=True))
      if dup_use_two_poles:
        is_not_dup_logit = tf.squeeze(tf.matmul(tf.matmul(h, self.is_not_dup_w), dup_min_arg_acc_h, transpose_b=True))
      else:
        is_not_dup_logit = tf.squeeze(tf.matmul(tf.matmul(h, self.is_not_dup_w), dup_max_arg_acc_h, transpose_b=True))
      result = tf.stack([is_not_dup_logit, is_dup_logit])
    elif is_dup_mode == mlp_is_dup:
      is_dup_logit = tf.squeeze(tf.matmul(tf.matmul(tf.concat([h, dup_max_arg_acc_h], axis=1), self.is_dup_w), self.is_dup_h, transpose_b=True))
      if dup_use_two_poles:
        is_not_dup_logit = tf.squeeze(tf.matmul(tf.matmul(tf.concat([h, dup_min_arg_acc_h], axis=1), self.is_not_dup_w), self.is_not_dup_h, transpose_b=True))
      else:
        is_not_dup_logit = tf.squeeze(tf.matmul(tf.matmul(tf.concat([h, dup_max_arg_acc_h], axis=1), self.is_not_dup_w), self.is_not_dup_h, transpose_b=True))
      result = tf.stack([is_not_dup_logit, is_dup_logit])
    elif is_dup_mode == sigmoid_is_dup:
      cared_h = dup_max_arg_acc_h
      if dup_use_two_poles:
        cared_h = tf.matmul(tf.concat([dup_max_arg_acc_h, dup_min_arg_acc_h], axis=1), self.two_poles_merge_w) + self.two_poles_merge_b
      is_dup_logit = tf.squeeze(tf.matmul(tf.concat([h, cared_h], axis=1), self.is_dup_h, transpose_b=True))
      result = is_dup_logit
    else:
      assert False, "Unrecognized is_dup_mode!"
    return result

  def compute_dup_loss(self, training, accumulated_en, oracle_en, oracle_relative, is_dup_logits, dup_logits, neg_dup_logit):
    total_length = tf.shape(dup_logits)[-1]
    pre_real_exist = tf.logical_and(oracle_relative > 0, oracle_relative <= total_length)
    pre_exist = tf.cast(pre_real_exist, int_type)
    specified_index = tf.stack([0, total_length - oracle_relative])[pre_exist]
    r_dup_logits = dup_logits
    if dup_use_two_poles:
      negative_specified_index = tf.stack([total_length+1-1, 0])[pre_exist]
      r_dup_logits = tf.concat([dup_logits, neg_dup_logit], axis=0)
    ''' compute dup '''
    ''' compute accurate '''
    if training:
      dup_mrr, dup_accurate = tf.constant(0.0, float_type), tf.zeros([len(top_ks)], float_type)
    else:
      dup_mrr, dup_accurate = compute_dup_accurate(accumulated_en, oracle_en, specified_index, dup_logits)
    ''' maximize the most likely '''
    dup_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[specified_index], logits=[r_dup_logits])
    p_op = tf.print(["shape of dup_losses:", tf.shape(dup_losses)])
    with tf.control_dependencies([p_op]):
      dup_loss = tf.reduce_sum(dup_losses)
    if dup_use_two_poles:
      neg_dup_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[negative_specified_index], logits=[r_dup_logits])
      dup_loss -= tf.reduce_sum(neg_dup_losses)
    r_val = 1.0
    r_dup_mrr = tf.stack([tf.constant(r_val, float_type), dup_mrr])[pre_exist]
    r_dup_accurate = tf.stack([tf.tile([tf.constant(r_val, float_type)], [len(top_ks)]), dup_accurate])[pre_exist]
    if dup_use_two_poles:
      r_dup_loss = dup_loss
    else:
      r_dup_loss = tf.stack([tf.constant(0.0, float_type), dup_loss])[pre_exist]
    ''' compute is_dup '''
    if is_dup_mode < sigmoid_is_dup:
      is_dup_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[pre_exist], logits=[is_dup_logits])
      predict_to_use_pre_exist = tf.cast(tf.argmax(is_dup_logits), int_type)
    else:
      is_dup_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=[tf.cast(pre_exist, float_type)], logits=[is_dup_logits])
      predict_to_use_pre_exist = tf.cast(is_dup_logits > 0.5, int_type)
    is_dup_accurate_one = tf.cast(tf.equal(predict_to_use_pre_exist, pre_exist), float_type)
    is_dup_loss = tf.reduce_sum(is_dup_losses)
    is_dup_accurate = tf.tile([is_dup_accurate_one], [len(top_ks)])
    ''' maximize the most likely '''
    total_mrr = tf.multiply(r_dup_mrr, is_dup_accurate_one)
    total_accurate = tf.multiply(r_dup_accurate, is_dup_accurate)
    total_loss = r_dup_loss + is_dup_loss
    return total_mrr, total_accurate, total_loss, r_dup_mrr, r_dup_accurate, predict_to_use_pre_exist


def compute_dup_accurate(oracle_token_sequence, oracle_type_content_en, specified_index, dup_logits):
  dup_size = tf.shape(dup_logits)[0]
  return tf.cond(dup_size > 1, lambda: compute_top_k_accurate(oracle_token_sequence, oracle_type_content_en, specified_index, dup_logits), lambda: (tf.constant(0.0, float_type), tf.zeros([len(top_ks)], float_type)))


def compute_top_k_accurate(oracle_token_sequence, oracle_type_content_en, specified_index, dup_logits):
  dup_size = tf.shape(dup_logits)[0]
  r_r = tf.minimum(dup_size, mrr_max)
  _, indices_r = tf.nn.top_k(dup_logits, r_r)
  if repetition_accuracy_mode == en_match:
    selected_tokens = tf.gather(oracle_token_sequence, indices_r)
    zero_one = tf.cast(tf.equal(oracle_type_content_en, selected_tokens), int_type)
  elif repetition_accuracy_mode == exact_accurate:
    zero_one = tf.cast(tf.equal(specified_index, indices_r), int_type)
  else:
    assert False, "Unrecognized repetition mode!"
  accs = tf.reduce_sum(zero_one)
  mrr = tf.cond(accs > 0, lambda: tf.math.divide(tf.constant(1.0, float_type), tf.cast(tf.argmax(zero_one) + 1, float_type)), lambda: tf.constant(0.0, float_type))
  result = tf.zeros([0], float_type)
  for i in range(len(top_ks)):
    k = top_ks[i]
    r_k = tf.minimum(k, dup_size)
    _, indices = tf.nn.top_k(dup_logits, r_k)
    if repetition_accuracy_mode == en_match:
      r_selected_tokens = tf.gather(oracle_token_sequence, indices)
      zero_one = tf.cast(tf.equal(oracle_type_content_en, r_selected_tokens), int_type)
    elif repetition_accuracy_mode == exact_accurate:
      zero_one = tf.cast(tf.equal(specified_index, indices), tf.int32)
    else:
      assert False, "Unrecognized repetition mode!"
    accs = tf.reduce_sum(zero_one)
    result = tf.concat([result, [tf.cast(accs > 0, float_type)]], axis=0)
  return mrr, result

