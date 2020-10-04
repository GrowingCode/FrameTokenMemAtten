import os
import shutil


def delete_file(f):
  if(os.path.exists(f)):
    os.remove(f)


def delete_directory(d):
  shutil.rmtree(d)


def clear_directory(d):
  ls = os.listdir(d)
  for i in ls:
    f_path = os.path.join(d, i)
    if os.path.isdir(f_path):
      clear_directory(f_path)
    os.remove(f_path)


def all_files_in_directory(d):
  result = []
  get_dir = os.listdir(d)
  for i in get_dir:
    sub_dir = os.path.join(d,i)
    if os.path.isdir(sub_dir):
      result.extend(all_files_in_directory(sub_dir))
    else:
      result.append(i)
  return result


def copy_files_from_one_directory_to_another_directory(d1, d2):
  d1_length = len(d1)
  files_in_d1 = all_files_in_directory(d1)
  for f1 in files_in_d1:
    f2 = d2 + d1[d1_length:]
    shutil.copy(f1, f2)
  




