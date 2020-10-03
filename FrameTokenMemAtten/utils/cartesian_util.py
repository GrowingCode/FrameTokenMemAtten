import tensorflow as tf
from metas.non_hyper_constants import float_type


def cartesian_add_one_dim_vector(vec1, vec2):
  sz = tf.shape(vec2)[0]
  
  e_vec1 = tf.expand_dims(vec1, 1)
  e_vec2 = tf.expand_dims(vec2, 0)
  
  oe_vec1 = tf.tile(e_vec1, [1, sz])
  oe_vec2 = tf.tile(e_vec2, [sz, 1])
  
  temp_vec = oe_vec1 + oe_vec2
  
  f_vec = tf.squeeze(tf.reshape(temp_vec, [sz * sz, 1]), axis=1)
  
  return f_vec


def cartesian_concat_two_dim_mats(vec1, vec2):
  cz = tf.shape(vec1)[1]
  sz = tf.shape(vec2)[0]
  
  e_mat1 = tf.reshape(tf.tile(vec1, [1, sz]), [-1, cz])
  e_mat2 = tf.tile(vec2, [sz, 1])
  
  f_mat = tf.concat([e_mat1, e_mat2], axis=1)
  
  return f_mat


if __name__ == '__main__':
  sess = tf.compat.v1.InteractiveSession()
  vec1 = tf.constant([10,20,30,40,50], float_type)
  vec2 = tf.constant([16,17,18,19,20], float_type)
  res_vec = cartesian_add_one_dim_vector(vec1, vec2)
  print(res_vec)
  res_mat = cartesian_concat_two_dim_mats(tf.tile(tf.expand_dims(vec1, axis=0), [5, 1]), tf.expand_dims(vec2, axis=1))
  print(res_mat)
  sess.close()





