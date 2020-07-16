import json
import numpy as np
from builtins import len
from metas.hyper_settings import top_ks
from metas.non_hyper_constants import np_float_type


top_rate = 0.5


def handle_one_example(example_index, each_noavg, each_count_noavg, info, size_info):
  one_example_noavg = each_noavg[example_index]
  one_example_noavg = np.array(one_example_noavg)
  one_example_r_count_noavg = each_count_noavg[example_index]
  one_example_r_count_noavg = np.array(one_example_r_count_noavg)
  
  noavg_size = np.shape(one_example_noavg)[0]
  count_noavg_size = np.shape(one_example_r_count_noavg)[0]
  assert noavg_size == count_noavg_size
  
  all_tokens = np.zeros((len(top_ks)), np_float_type)
  all_count = 0
  for index in range(noavg_size):
    one_token = one_example_noavg[index]
    one_count = one_example_r_count_noavg[index]
    if one_count > 0:
      all_tokens += one_token
      all_count += 1
  
  avg = all_tokens / all_count
  info.append(avg)
  size_info.append(all_count)


'''
mode
0: sort by (1st arg)
1: sort by (1st arg / 2nd arg)
'''
def analyze_one_set(json_path, compare_json_path=None, mode=0):
  with open(json_path, 'r') as json_record:
    record_json = json.load(json_record)
    each_noavg = record_json["token_accurate_each_noavg"]
    each_count_noavg = record_json["token_count_each_int_noavg"]
  compare_each_noavg = None
  compare_each_count_noavg = None
  if compare_json_path != None:
    with open(compare_json_path, 'r') as compare_json_record:
      compare_record_json = json.load(compare_json_record)
      compare_each_noavg = compare_record_json["token_accurate_each_noavg"]
      compare_each_count_noavg = record_json["token_count_each_int_noavg"]
  
  quota = []
  info = []
  size_info = []
  if compare_json_path:
    compare_info = []
    compare_size_info = []
  
  num_of_examples = len(each_noavg)
  for index in range(num_of_examples):
    if compare_json_path:
      assert each_count_noavg[index] <= compare_each_count_noavg[index], "compare count must be no-filtered or less-filtered"
    
    handle_one_example(index, each_noavg, each_count_noavg, info, size_info)
    if compare_json_path:
      handle_one_example(index, compare_each_noavg, compare_each_count_noavg, compare_info, compare_size_info)
    
    for idx in range(len(info)):
      if mode == 0:
        quota.append(info[idx][0])
      elif mode == 1:
        assert compare_json_path
        quota.append(info[idx][0] / compare_info[idx][0])
      else:
        assert False
    
  np_quota = np.array(quota)
  sorted_index = np.argsort(-np_quota)
  res = []
  for index in range(num_of_examples):
    s_idx = sorted_index[index]
    o_r = [s_idx, np_quota[s_idx], info[s_idx], size_info[s_idx]]
    if compare_json_path:
      o_r.append(compare_info[s_idx])
      o_r.append(compare_size_info[s_idx])
    res.append(o_r)
#     print("example_index:" + str(s_idx) + "#" + str(info[s_idx]) + "#total_size:" + str(size_info[s_idx]))
  return res
  
    
def analyze_two_sets(json_path1, json_path2, compare_json_path1=None, compare_json_path2=None, mode=0):
  res1 = analyze_one_set(json_path1, compare_json_path1, mode)
  res2 = analyze_one_set(json_path2, compare_json_path2, mode)
  res1_len = len(res1)
  res2_len = len(res2)
  all_size = res1_len + res2_len
  r_size = int(all_size * top_rate)
  new_res = []
  idx1 = 0
  idx2 = 0
  while len(new_res) < r_size:
    if idx1 < res1_len:
      r1 = res1[idx1][1]
    else:
      r1 = 0.0
    if idx2 < res2_len:
      r2 = res2[idx2][1]
    else:
      r2 = 0.0
    if r1 < r2:
      r = res2[idx2]
      idx2 += 1
    else:
      r = res1[idx1]
      idx1 += 1
    new_res.append(r)
  assert len(new_res) == r_size
  f_tk = np.array([0.0, 0.0, 0.0, 0.0])
  f_size = 0.0
  if compare_json_path1:
    c_f_tk = np.array([0.0, 0.0, 0.0, 0.0])
    c_f_size = 0.0
  for i in range(r_size):
    f_tk += new_res[i][2] * new_res[i][3]
    f_size += new_res[i][3]
    if compare_json_path1:
      c_f_tk += new_res[i][4] * new_res[i][5]
      c_f_size += new_res[i][5]
  f_res = f_tk / f_size
  c_f_res = None
  if compare_json_path1:
    c_f_res = c_f_tk / c_f_size
  print("final_result:" + str(f_res) + "#" + str(c_f_res))
  
  
  
  
if __name__ == '__main__':
  analyze_two_sets("D:/HomeSpace/paper-workspace/REP/Experiment/simple-sequence_token_decode/zoot_run_info_record/valid_noavg.json", "D:/HomeSpace/paper-workspace/REP/Experiment/simple-sequence_token_decode/zoot_run_info_record/test_noavg.json")
  analyze_two_sets("C:/Users/yangy/Desktop/Experiment/findbugs/1000unk-5stop-specified_fold-statement_rep_dup_sequence_only_memory_style/simplename/zoot_run_info_record/valid_noavg.json", "C:/Users/yangy/Desktop/Experiment/findbugs/1000unk-5stop-specified_fold-statement_rep_dup_sequence_only_memory_style/simplename/zoot_run_info_record/test_noavg.json")
  
  
  








