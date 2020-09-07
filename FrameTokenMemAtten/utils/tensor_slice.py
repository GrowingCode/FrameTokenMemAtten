import tensorflow as tf


def extract_subsequence_with_start_end_info(seq, start, end):
  t_seq = tf.slice(seq, start, end - start + 1)
  return t_seq
  



