import tensorflow as tf


def make_sure_shape_of_tensor_array(t_array, sure_size):
  
  def s_cond(i, *_):
    return tf.less(i, sure_size)
  
  def s_body(i, t_array):
    t_array = t_array.write(i, 0)
    return i+1, t_array
  
  _, t_array = tf.while_loop(s_cond, s_body, [t_array.size(), t_array], shape_invariants=[tf.TensorShape(()), tf.TensorShape(None)], parallel_iterations=1)
  
  return t_array

