import os
import shutil
from metas.non_hyper_constants import model_storage_dir, model_config,\
  model_storage_parent_dir


def check_or_store_configs():
  check_or_copy_one_config('../metas/hyper_settings.py')


def check_or_copy_one_config(origin_config_file):
  real_model_storage_dir = '../' + model_storage_parent_dir + '/' + model_storage_dir
  if not os.path.exists(real_model_storage_dir):
    os.makedirs(real_model_storage_dir)
    
  name1 = origin_config_file
  name2 = real_model_storage_dir + '/' + model_config
  if os.path.exists(name2):
    assert files_are_same(name1, name2), "Configuration Inconsistent!"
  else:
    shutil.copyfile(name1, name2)
  
  
def files_are_same(name1, name2):
  with open(name1) as f1:
    with open(name2) as f2:
      f1_lines = f1.readlines()
      f2_lines = f2.readlines()
      f1_len = len(f1_lines)
      f2_len = len(f2_lines)
      if f1_len == f2_len:
        for i in range(f1_len):
          line1 = f1_lines[i]
          line2 = f2_lines[i]
          if(line1 != line2):
            return False
      else:
        return False
  return True





