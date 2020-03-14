import tensorflow as tf
import numpy as np


def convert_numpy_to_tensor(np_arrays):
  result = []
  for np_array in np_arrays:
    if isinstance(np_array, list) or isinstance(np_array, tuple):
      result.append(convert_numpy_to_tensor(np_array))
    else:
      assert type(np_array) is np.ndarray
      result.append(tf.convert_to_tensor(np_array))
  return result





