import tensorflow as tf
from metas.non_hyper_constants import float_type


# a = tf.ones([3,3,3], float_type)
a = tf.ones([3,3], float_type)
# a = tf.ones([5], float_type)

b = a[:,None,:]
b_shape = tf.shape(b)

with tf.Session() as sess:
  res = sess.run([b, b_shape])
  print(res)



