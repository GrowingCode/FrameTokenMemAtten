

def get_content_between_two_specified_string(o_str, b_str, e_str):
  b_idx = o_str.find(b_str)
  e_idx = o_str.find(e_str)
  sp = o_str[b_idx+1:e_idx]
  return sp


if __name__ == "__main__":
  print(get_content_between_two_specified_string("a:12#:34#", ":", "#"))




