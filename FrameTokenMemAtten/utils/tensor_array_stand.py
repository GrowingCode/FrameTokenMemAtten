from metas.non_hyper_constants import max_threshold_example_length, np_int_type,\
  np_float_type
import tensorflow as tf


invalid_mark = np_int_type(-12345678)

def make_sure_shape_of_tensor_array(t_array):
  
  def s_cond(i, *_):
    return tf.less(i, max_threshold_example_length)
  
  def s_body(i, t_array):
    t_array = t_array.write(i, invalid_mark)
    return i+1, t_array
  
  _, t_array = tf.while_loop(s_cond, s_body, [t_array.size(), t_array], shape_invariants=[tf.TensorShape(()), tf.TensorShape(None)], parallel_iterations=1)
  
  return t_array


def convert_tensor_array_to_lists_of_tensors(t_array):
  t_array = make_sure_shape_of_tensor_array(t_array)
  tensors = [t_array.read(i) for i in range(max_threshold_example_length)]
  return tensors


def filter_out_invalid_mark(np_list):
  s_idx = 0
  for ele_idx, nl in enumerate(np_list):
    is_number = type(nl) is np_int_type or type(nl) is np_float_type
    equal_invalid = nl == invalid_mark
    if is_number and equal_invalid:
      s_idx = ele_idx
      break;
#     print("equal_invalid:" + str(equal_invalid) + "#is_number:" + str(is_number) + "#ele_idx:" + str(ele_idx) + "#nl:" + str(nl) + "#")
    
#   happen = np_list.count(invalid_mark) > 0
#   if happen:
#     m_idx = np_list.index(invalid_mark)
  np_list = np_list[0:s_idx]
#     print("m_idx:" + str(m_idx))
  return np_list


# filter_out_invaid_mark([1, 2, 4, invalid_mark, 100, invalid_mark])




