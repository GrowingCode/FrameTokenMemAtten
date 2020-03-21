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


def build_statement_feed_dict(line):
  raw_sequence_decode_str = line.strip()
  sequence_decode_str = raw_sequence_decode_str.split("$")[0]
  strs = sequence_decode_str.split("#")
#   inner_index_type_content_index_str = strs[0]
#   inner_index_type_content_index_str = inner_index_type_content_index_str.split()
#   inner_index_type_content_index = [int(inner_d) for inner_d in inner_index_type_content_index_str]
  
  i = 0
  i_len = len(strs)
  sequence_decodes_np_array = np.zeros([0, len(strs[0].split())], np_int_type)
  while (i < 7):
    one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
#     print("np.shape(sequence_decodes_np_array):" + str(np.shape(sequence_decodes_np_array)) + "#np.shape(one_dim):" + str(np.shape(one_dim)))
    sequence_decodes_np_array = np.concatenate((sequence_decodes_np_array, one_dim), axis=0)
    i = i + 1
  assert i == 7
  
  sequence_decodes_start = [int(id_str) for id_str in strs[i].split()]
  i = i + 1
  sequence_decodes_end = [int(id_str) for id_str in strs[i].split()]
  i = i + 1
  assert i == 9
  
#   '''
#   variables in statement
#   '''
#   stmt_variable_info = np.zeros([0, len(strs[i].split())], np_int_type)
#   while (i < 11):
#     one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
# #     print("stmt_variable_info:" + str(np.shape(stmt_variable_info)) + "#one_dim:" + str(np.shape(one_dim)))
#     stmt_variable_info = np.concatenate([stmt_variable_info, one_dim], axis=0)
#     i = i + 1
#   stmt_variable_info_start = [int(id_str) for id_str in strs[i].split()]
#   i = i + 1
#   stmt_variable_info_end = [int(id_str) for id_str in strs[i].split()]
#   i = i + 1
  
  '''
  following statements
  '''
  stmt_following_legal_info = [int(id_str) for id_str in strs[i].split()]
  i = i + 1
  stmt_following_legal_info_start = [int(id_str) for id_str in strs[i].split()]
  i = i + 1
  stmt_following_legal_info_end = [int(id_str) for id_str in strs[i].split()]
  i = i + 1
  
#   '''
#   swords
#   '''
#   assert i == 16
#   one_swords = np.zeros([0, len(strs[i].split())], np_int_type)
#   while (i < 19):
#     one_dim = [[int(id_str) for id_str in strs[i].split()]]
#     one_swords = np.concatenate((one_swords, one_dim), axis=0)
#     i = i + 1
#   one_token_sword_start = [int(id_str) for id_str in strs[i].split()]
#   i = i + 1
#   one_token_sword_end = [int(id_str) for id_str in strs[i].split()]
#   i = i + 1
#   
#   '''
#   sword_variables in statement
#   '''
#   assert i == 21
#   stmt_sword_variable_info = np.zeros([0, len(strs[i].split())], np_int_type)
#   while (i < 23):
#     one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
#     stmt_sword_variable_info = np.concatenate([stmt_sword_variable_info, one_dim], axis=0)
#     i = i + 1
#   stmt_sword_variable_info_start = [int(id_str) for id_str in strs[i].split()]
#   i = i + 1
#   stmt_sword_variable_info_end = [int(id_str) for id_str in strs[i].split()]
#   i = i + 1
  assert i == i_len
  
  '''
  sequence_decodes: contain 2 rows:
  0th row: type_content id
  1th row: decode type: 0 do normally, 1 decode until nature
  '''
#   sequence_decodes = [int(ad) for ad in sequence_decode_str.split()]
#   sequence_decodes = np.array(sequence_decodes)
#   len_sequence_decodes = np.shape(sequence_decodes)[0]
#   sequence_decodes = np.reshape(sequence_decodes, (sequence_decode_first_dimension, int(len_sequence_decodes / sequence_decode_first_dimension)))
  '''
  build feed_dict with tensors
  '''
#   sequence_decodes_np_array = sequence_decodes
#   np.count_nonzero
  validate_relatives(sequence_decodes_np_array, "token")
#   validate_relatives(one_swords, "sword")
#   np.shape(one_swords)[-1], np.shape(sequence_decodes_np_array)[-1], np.count_nonzero(sequence_decodes_np_array[2] >= 0), 
#   one_swords, one_token_sword_start, one_token_sword_end, 
#   , stmt_sword_variable_info, stmt_sword_variable_info_start, stmt_sword_variable_info_end
# stmt_variable_info, stmt_variable_info_start, stmt_variable_info_end, 
  return (sequence_decodes_np_array[0:3,:], sequence_decodes_start, sequence_decodes_end, stmt_following_legal_info, stmt_following_legal_info_start, stmt_following_legal_info_end)

  
# def get_size_of_one_sequence_example(one_example):
#   return one_example[0]
def validate_relatives(one_swords, e_info):
  i_len = np.shape(one_swords)[-1]
  for i in range(i_len):
    assert one_swords[1][i] <= i+1, e_info + "_sequence_length:" + str(i_len) + "#one_swords[1][i]:" + str(one_swords[1][i]) + "#i:" + str(i)





