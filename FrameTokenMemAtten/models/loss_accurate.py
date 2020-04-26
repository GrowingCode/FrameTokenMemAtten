import tensorflow as tf
from metas.non_hyper_constants import float_type, int_type
from metas.hyper_settings import top_ks, mrr_max


def compute_logits_given_to_deocde_embed_with_computed_embeddings(computed_embeddings, to_decode_embed):
  processed_to_decode_embed = to_decode_embed
  logits_raw = tf.matmul(processed_to_decode_embed, computed_embeddings, transpose_b=True)
  logits = tf.squeeze(logits_raw, axis=[0])
  return logits


def linear_loss(oracle_type_content_en, logits):# desc_prefix, type_content_data, 
  specified_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=oracle_type_content_en, logits=logits)
  return specified_loss


def compute_top_n(logits, n):
  length_of_logits = tf.shape(logits)[0]
  r_r = tf.minimum(length_of_logits, n)
  _, tokens = tf.nn.top_k(logits, r_r)
  return tokens


def compute_linear_accurate(oracle_type_content_en, logits):
  length_of_logits = tf.shape(logits)[0]
  r_r = tf.minimum(length_of_logits, mrr_max)
  _, indices_r = tf.nn.top_k(logits, r_r)
  zero_one = tf.cast(tf.equal(oracle_type_content_en, indices_r), int_type)
  accs = tf.reduce_sum(zero_one)
  mrr = tf.stack([tf.constant(0.0, float_type), tf.divide(tf.constant(1.0, float_type), tf.cast(tf.argmax(zero_one)+1, float_type))])[tf.cast(accs > 0, int_type)]
  accurate = tf.zeros([0], float_type)
  for i in range(len(top_ks)):
    k = top_ks[i]
    r_k = tf.minimum(length_of_logits, k)
    _, indices = tf.nn.top_k(logits, r_k)
    zero_one = tf.cast(tf.equal(oracle_type_content_en, indices), int_type)
    accs = tf.reduce_sum(zero_one)
    accurate = tf.concat([accurate, [tf.cast(accs > 0, float_type)]], axis=0)
  return mrr, accurate
  

def compute_loss_and_accurate_from_linear_with_computed_embeddings(training, computed_embeddings, oracle_index_in_computed_embeddings, output):
  '''
  public class for computing loss and accurate
  '''
  logits = compute_logits_given_to_deocde_embed_with_computed_embeddings(computed_embeddings, output)
  loss = linear_loss(oracle_index_in_computed_embeddings, logits)# desc_prefix, type_content_data, 
  '''
  the following two functions are to compute accurate
  compute accuracy for type and content
  '''
  if training:
    mrr, accurate = tf.constant(0.0, float_type), tf.zeros([len(top_ks)], float_type)
  else:
    mrr, accurate = compute_linear_accurate(oracle_index_in_computed_embeddings, logits)
  return mrr, accurate, loss


def compute_loss_and_accurate_from_linear_with_computed_embeddings_in_limited_range(training, computed_embeddings, ens_in_range, en, output):
  '''
  public class for computing loss and accurate
  '''
  computed_embeddings_in_range = tf.gather(computed_embeddings, ens_in_range)
  indexes = tf.where(tf.equal(en, ens_in_range))
  indexes_len = tf.shape(indexes)[0]
  exist = tf.cast(tf.greater(indexes_len, 0), int_type)
  indexes = tf.concat([indexes, [[0]]], axis=0)
  oracle_index = tf.cast(indexes[0][0], int_type)
  logits = compute_logits_given_to_deocde_embed_with_computed_embeddings(computed_embeddings_in_range, output)
#   p_op = tf.print(["shape_of_logits:", tf.shape(logits), "indexes:", indexes, "oracle_index:", oracle_index, "ens_in_range:", ens_in_range, "shape_computed_embeddings_in_range:", tf.shape(computed_embeddings_in_range)])
#   with tf.control_dependencies([p_op]):
  loss = linear_loss(oracle_index, logits)
  '''
  the following two functions are to compute accurate
  compute accuracy for type and content
  '''
  if training:
    mrr, accurate = tf.constant(0.0, float_type), tf.zeros([len(top_ks)], float_type)
  else:
    mrr, accurate = compute_linear_accurate(oracle_index, logits)
  
  mrr_stand, accurate_stand, loss_stand = compute_loss_and_accurate_from_linear_with_computed_embeddings(training, computed_embeddings, en, output)
  
  r_mrr = tf.stack([mrr_stand, mrr])[exist]
  r_accurate = tf.stack([accurate_stand, accurate])[exist]
  r_loss = tf.stack([loss_stand, loss])[exist]
  
  return r_mrr, r_accurate, r_loss



