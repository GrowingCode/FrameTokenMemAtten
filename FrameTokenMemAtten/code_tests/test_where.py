import tensorflow as tf


en, ens_in_range = 3, [1, 2, 3, 4, 5, 2]

a = tf.where(tf.equal(en, ens_in_range))
before_a_shape = tf.shape(a)
indexes_len = tf.shape(a)[0]
exist = tf.cast(tf.greater(indexes_len, 0), tf.int32)
a = tf.concat([a, [[0]]], axis=0)
oracle_index = a[0][0]

after_a_shape = tf.shape(a)

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
  a_v, before_a_shape_v, after_a_shape_v, exist_v, oracle_index_v = sess.run([a, before_a_shape, after_a_shape, exist, oracle_index])
  print(a_v)
  print(before_a_shape_v)
  print(after_a_shape_v)
  print(exist_v)
  print(oracle_index_v)

