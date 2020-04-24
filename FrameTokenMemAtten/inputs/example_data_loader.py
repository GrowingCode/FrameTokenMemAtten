import numpy as np
from metas.non_hyper_constants import data_dir, np_int_type


def build_skeleton_feed_dict(mode):
  pfx = "skt"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", ) as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    raw_sequence_decode_str = line.strip()
    sequence_decode_str = raw_sequence_decode_str
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
    
    examples.append((stmt_skeleton_token_info, stmt_skeleton_token_info_start, stmt_skeleton_token_info_end))
  
  return examples


def build_statement_feed_dict(mode):
  pfx = "stmt"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", ) as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    raw_sequence_decode_str = line.strip()
    sequence_decode_str = raw_sequence_decode_str
    strs = sequence_decode_str.split("#")

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
    
    '''
    following statements
    '''
    stmt_following_legal_info = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    stmt_following_legal_info_start = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    stmt_following_legal_info_end = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    
    assert i == i_len
    
    '''
    sequence_decodes: contain 2 rows:
    0th row: type_content id
    1th row: decode type: 0 do normally, 1 decode until nature
    '''
    '''
    build feed_dict with tensors
    '''
    validate_relatives(sequence_decodes_np_array, "token")
    examples.append((sequence_decodes_np_array[0:3,:], sequence_decodes_start, sequence_decodes_end, stmt_following_legal_info, stmt_following_legal_info_start, stmt_following_legal_info_end))
  return examples
  
  
def build_sequence_feed_dict(mode):
  pfx = "sequence"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", ) as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    raw_sequence_decode_str = line.strip()
    sequence_decode_str = raw_sequence_decode_str
    strs = sequence_decode_str.split("#")
    
    i = 0
    seq_length = len(strs[0].split())
    sequence_token_info = np.zeros([0, seq_length], np_int_type)
    while (i < 1):
      one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
      sequence_token_info = np.concatenate((sequence_token_info, one_dim), axis=0)
      i = i + 1
    
    assert i == 1
    
    examples.append((sequence_token_info, np.array([0]), np.array([seq_length-1])))
  
  return examples


def build_tree_feed_dict(mode):
  pfx = "tree"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", ) as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    tree_decode_str = line.strip()
    strs = tree_decode_str.split("#")
    
    i = 0
    post_order_node_type_content_en = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    
    post_order_node_child_start = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    post_order_node_child_end = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    post_order_node_children = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    
    pre_post_order_node_type_content_en = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    pre_post_order_node_state = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    pre_post_order_node_post_order_index = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    pre_post_order_node_parent_grammar_index = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    
    assert i == 8
    
    examples.append((post_order_node_type_content_en, post_order_node_child_start, post_order_node_child_end, post_order_node_children, pre_post_order_node_type_content_en, pre_post_order_node_state, pre_post_order_node_post_order_index, pre_post_order_node_parent_grammar_index))
  
  return examples
  
  
def validate_relatives(one_swords, e_info):
  i_len = np.shape(one_swords)[-1]
  for i in range(i_len):
    assert one_swords[1][i] <= i+1, e_info + "_sequence_length:" + str(i_len) + "#one_swords[1][i]:" + str(one_swords[1][i]) + "#i:" + str(i)





