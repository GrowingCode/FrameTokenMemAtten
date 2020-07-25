import json
import os

from metas.hyper_settings import dup_base_model_directory
from metas.non_hyper_constants import data_dir, np_int_type,\
  min_threshold_example_length, max_threshold_example_length
import numpy as np


def build_skeleton_dup_feed_dict(mode):
  examples = build_skeleton_feed_dict(mode)
  return build_dup_feed_dict(mode, examples)


def build_skeleton_feed_dict(mode):
  pfx = "skt"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", 'r') as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    raw_sequence_decode_str = line.strip()
    sequence_decode_str = raw_sequence_decode_str
    strs = sequence_decode_str.split("#")
    
    i = 0
    i_len = len(strs)
    e_len = len(strs[0].split())
    if (e_len < min_threshold_example_length or e_len > max_threshold_example_length):
      continue;
    stmt_skeleton_token_info = np.zeros([0, e_len], np_int_type)
    while (i < 5):
      one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
      stmt_skeleton_token_info = np.concatenate((stmt_skeleton_token_info, one_dim), axis=0)
      i = i + 1
    
    stmt_skeleton_token_info_start = np.array([int(id_str) for id_str in strs[i].split()])
    i = i + 1
    stmt_skeleton_token_info_end = np.array([int(id_str) for id_str in strs[i].split()])
    i = i + 1
    stmt_skeleton_token_struct_info_end = np.array([int(id_str) for id_str in strs[i].split()])
    i = i + 1
    
    assert i == i_len
    
    examples.append((stmt_skeleton_token_info, stmt_skeleton_token_info_start, stmt_skeleton_token_info_end, stmt_skeleton_token_struct_info_end))
  
  return examples


def build_statement_dup_feed_dict(mode):
  examples = build_statement_feed_dict(mode)
  return build_dup_feed_dict(mode, examples)


def build_statement_feed_dict(mode):
  pfx = "stmt"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", 'r') as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    raw_sequence_decode_str = line.strip()
    sequence_decode_str = raw_sequence_decode_str
    strs = sequence_decode_str.split("#")

    i = 0
    i_len = len(strs)
    e_len = len(strs[0].split())
    if (e_len < min_threshold_example_length or e_len > max_threshold_example_length):
      continue;
    sequence_decodes_np_array = np.zeros([0, e_len], np_int_type)
    while (i < 5):
      one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
  #     print("np.shape(sequence_decodes_np_array):" + str(np.shape(sequence_decodes_np_array)) + "#np.shape(one_dim):" + str(np.shape(one_dim)))
      sequence_decodes_np_array = np.concatenate((sequence_decodes_np_array, one_dim), axis=0)
      i = i + 1
    assert i == 5
    
    sequence_decodes_start = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    sequence_decodes_end = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    assert i == 7
    
#     '''
#     following statements
#     '''
#     stmt_following_legal_info = [int(id_str) for id_str in strs[i].split()]
#     i = i + 1
#     stmt_following_legal_info_start = [int(id_str) for id_str in strs[i].split()]
#     i = i + 1
#     stmt_following_legal_info_end = [int(id_str) for id_str in strs[i].split()]
#     i = i + 1
    
    assert i == i_len, "i:" + str(i) + "i_len:" + str(i_len)
    
    '''
    sequence_decodes: contain 2 rows:
    0th row: type_content id
    1th row: decode type: 0 do normally, 1 decode until nature
    '''
    '''
    build feed_dict with tensors
    '''
    validate_relatives(sequence_decodes_np_array, "token")
#     , stmt_following_legal_info, stmt_following_legal_info_start, stmt_following_legal_info_end
    examples.append((sequence_decodes_np_array, sequence_decodes_start, sequence_decodes_end))
  return examples


def build_linear_dup_feed_dict(mode):
  examples = build_sequence_feed_dict(mode)
  return build_dup_feed_dict(mode, examples)

  
def build_sequence_feed_dict(mode):
  pfx = "sequence"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", 'r') as data_file:
    raw_datas = data_file.readlines()
  
  examples = []
  
  for line in raw_datas:
    raw_sequence_decode_str = line.strip()
    sequence_decode_str = raw_sequence_decode_str
    strs = sequence_decode_str.split("#")
    
    i = 0
    seq_length = len(strs[0].split())
    if (seq_length < min_threshold_example_length or seq_length > max_threshold_example_length):
      continue;
    sequence_token_info = np.zeros([0, seq_length], np_int_type)
    while (i < 5):
      one_dim = np.array([[int(id_str) for id_str in strs[i].split()]])
      sequence_token_info = np.concatenate((sequence_token_info, one_dim), axis=0)
      i = i + 1
    
    assert i == 5
    
    examples.append((sequence_token_info, np.array([0]), np.array([seq_length-1])))
  
  return examples


def build_tree_feed_dict(mode):
  pfx = "tree"
  with open(data_dir + "/" + pfx + "_" + mode + "_data.txt", 'r') as data_file:
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
    e_len = len(pre_post_order_node_state) - sum(np.array(pre_post_order_node_state) == 2)
    if (e_len < min_threshold_example_length or e_len > max_threshold_example_length):
      continue;
    i = i + 1
    
    pre_post_order_node_post_order_index = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    pre_post_order_node_parent_grammar_index = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    pre_post_order_node_kind = [int(id_str) for id_str in strs[i].split()]
    i = i + 1
    
    assert i == 9
    
    examples.append((post_order_node_type_content_en, post_order_node_child_start, post_order_node_child_end, post_order_node_children, pre_post_order_node_type_content_en, pre_post_order_node_state, pre_post_order_node_post_order_index, pre_post_order_node_parent_grammar_index, pre_post_order_node_kind))
  
  return examples
  
  
def build_dup_feed_dict(mode, examples):
  f_path = dup_base_model_directory + "/" + mode + "_noavg.json"
  assert os.path.exists(f_path), dup_base_model_directory + " contains no such " + mode + "_noavg.json"
  with open(f_path, 'r') as turn_record:
    turn_record_json = json.load(turn_record)
  accuracy_noavg = np.array(turn_record_json["token_accurate_each_noavg"])
  mrr_noavg = np.array(turn_record_json["token_mrr_each_noavg"])
  assert len(examples) == len(accuracy_noavg) and len(accuracy_noavg) == len(mrr_noavg)
  new_examples = []
  for example, accuracy, mrr in zip(examples, accuracy_noavg, mrr_noavg):
    new_examples.append([example, accuracy, mrr])
  return new_examples
  
  
def validate_relatives(one_swords, e_info):
  i_len = np.shape(one_swords)[-1]
  for i in range(i_len):
    assert one_swords[1][i] <= i+1, e_info + "_sequence_length:" + str(i_len) + "#one_swords[1][i]:" + str(one_swords[1][i]) + "#i:" + str(i)





