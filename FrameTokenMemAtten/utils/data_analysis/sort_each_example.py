import json
import numpy as np
from builtins import len


'''
mode
0: sort by (1st arg)
1: sort by (1st arg / 2nd arg)
'''
def analyze_one_set(json_path, compare_json_path=None, mode=0):
  with open(json_path, 'r') as json_record:
    record_json = json.load(json_record)
    each_noavg = record_json["token_accurate_each_noavg"]
  if compare_json_path != None:
    with open(compare_json_path, 'r') as compare_json_record:
      compare_record_json = json.load(compare_json_record)
      compare_each_noavg = compare_record_json["token_accurate_each_noavg"]
  
  quota = []
  info = []
  size_info = []
  if compare_json_path:
    compare_info = []
    compare_size_info = []
  
  num_of_examples = len(each_noavg)
  for index in range(num_of_examples):
    one_example_noavg = each_noavg[index]
    one_example_noavg = np.array(one_example_noavg)
    noavg_size = np.shape(one_example_noavg)[0]
    avg = np.sum(one_example_noavg, axis=0) / noavg_size
    info.append(avg)
    size_info.append(noavg_size)
    if compare_json_path:
      compare_one_example_noavg = compare_each_noavg[index]
      compare_one_example_noavg = np.array(compare_one_example_noavg)
      compare_noavg_size = np.shape(compare_one_example_noavg)[0]
      compare_avg = np.sum(compare_one_example_noavg, axis=0) / compare_noavg_size
      compare_info.append(compare_avg)
      compare_size_info.append(compare_noavg_size)
    
    if mode == 0:
      quota.append(avg[0])
    elif mode == 1:
      quota.append(avg[0] / compare_avg[0])
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
  all_size = len(res1) + len(res2)
  r_size = all_size / 2
  new_res = []
  idx1 = 0
  idx2 = 0
  while len(new_res) < r_size:
    r1 = res1[idx1][0]
    r2 = res2[idx2][0]
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
  for i in range(int(r_size)):
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
  
  








