import numpy as np
from metas.non_hyper_constants import np_int_type


def build_skeleton_feed_dict(line):
  raw_sequence_decode_str = line.strip()
  sequence_decode_str = raw_sequence_decode_str.split("$")[2]
  strs = sequence_decode_str.split("#")
  
  i = 0
  i_len = len(strs)
  stmt_skeleton_token_info = np.zeros([0, len(strs[0].split())], np_int_type)
  while (i < 3):
    one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
    stmt_skeleton_token_info = np.concatenate((stmt_skeleton_token_info, one_dim), axis=0)
    i = i + 1
  stmt_skeleton_token_info_start = np.array([int(id_str) for id_str in strs[i].split()])
  i = i + 1
  stmt_skeleton_token_info_end = np.array([int(id_str) for id_str in strs[i].split()])
  i = i + 1
  
  assert i == i_len
  
  return (stmt_skeleton_token_info, stmt_skeleton_token_info_start, stmt_skeleton_token_info_end)


