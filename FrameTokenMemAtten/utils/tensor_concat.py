import tensorflow as tf
from metas.non_hyper_constants import int_type
from metas.hyper_settings import num_units


def concat_in_fixed_length_one_dimension(concat_er, c_one, c_size):
  concat_er = tf.concat([concat_er, c_one], axis=0)
  length_0 = tf.shape(concat_er)[0]
  over_size = length_0 - c_size
  slice_start = tf.cast(tf.greater(over_size, 0), int_type) * over_size
  concat_er = tf.slice(concat_er, [slice_start], [length_0-slice_start])
  return concat_er


def concat_in_fixed_length_two_dimension(concat_er, c_one, c_size):
  concat_er = tf.concat([concat_er, c_one], axis=0)
  length_0 = tf.shape(concat_er)[0]
  over_size = length_0 - c_size
  slice_start = tf.cast(tf.greater(over_size, 0), int_type) * over_size
  concat_er = tf.slice(concat_er, [slice_start, 0], [length_0-slice_start, num_units])
  return concat_er


