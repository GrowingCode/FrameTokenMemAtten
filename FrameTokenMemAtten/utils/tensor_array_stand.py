import tensorflow as tf
from metas.non_hyper_constants import max_threshold_example_length


def make_sure_shape_of_tensor_array(t_array):
  
  def s_cond(i, *_):
    return tf.less(i, max_threshold_example_length)
  
  def s_body(i, t_array):
    t_array = t_array.write(i, 0)
    return i+1, t_array
  
  _, t_array = tf.while_loop(s_cond, s_body, [t_array.size(), t_array], shape_invariants=[tf.TensorShape(()), tf.TensorShape(None)], parallel_iterations=1)
  
  return t_array


def convert_tensor_array_to_lists_of_tensors(t_array):
  tensors = [t_array.read(i) for i in range(max_threshold_example_length)]
  return tensors







