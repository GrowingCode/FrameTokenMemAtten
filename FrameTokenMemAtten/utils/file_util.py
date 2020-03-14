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








