from metas.hyper_settings import num_units
from metas.non_hyper_constants import float_type
import tensorflow as tf


zero_tensor = tf.zeros([1, num_units], dtype=float_type)

