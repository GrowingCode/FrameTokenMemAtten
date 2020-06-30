import json
import numpy as np


def analyze_one_set(json_path):
  with open(json_path, 'r') as json_record:
    record_json = json.load(json_record)
    each_noavg = record_json["token_accurate_each_noavg"]
  
  quota = []
  info = []
  size_info = []
  num_of_examples = len(each_noavg)
  for index in range(num_of_examples):
    one_example_noavg = each_noavg[index]
    one_example_noavg = np.array(one_example_noavg)
    noavg_size = np.shape(one_example_noavg)[0]
    avg = np.sum(one_example_noavg, axis=0) / noavg_size
    quota.append(avg[0])
    info.append(avg)
    size_info.append(noavg_size)
  np_quota = np.array(quota)
  sorted_index = np.argsort(-np_quota)
  for index in range(num_of_examples):
    s_idx = sorted_index[index]
    print("example_index:" + str(s_idx) + "#" + str(info[s_idx]) + "#total_size:" + str(size_info[s_idx]))
  
  
if __name__ == '__main__':
  analyze_one_set("D:/HomeSpace/paper-workspace/REP/Experiment/simple-sequence_token_decode/zoot_run_info_record/test_noavg.json")
  
  








