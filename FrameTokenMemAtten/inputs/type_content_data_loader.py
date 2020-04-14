import json
import tensorflow as tf
from metas.non_hyper_constants import all_token_summary,\
  SkeletonNum, TokenNum,\
  MaximumStringLength, all_token_char_sequences,\
  all_token_each_char_sequence_start, all_token_each_char_sequence_end,\
  all_token_each_subword_sequence_end, all_token_each_subword_sequence_start,\
  all_token_subword_sequences, data_dir, SkeletonHitNum, TokenHitNum, SwordNum,\
  SwordHitNum, CharNum, CharHitNum, all_subword_is_start_end


def load_type_content_data(type_content_data):
  
  all_token_summary_file = open(data_dir + "/All_token_summary.json", 'r', encoding='UTF-8')
  all_token_summary_ts = json.load(all_token_summary_file)
  all_token_summary_file.close()
  type_content_data[all_token_summary] = all_token_summary_ts
  
#   all_skeleton_id_file = open(data_dir + "/All_skeleton_id.json", 'r', encoding='UTF-8')
#   all_skeleton_id_ts = json.load(all_skeleton_id_file)
#   all_skeleton_id_string_map = {}
#   for all_skeleton_id_string_key in all_skeleton_id_ts:
#     all_skeleton_id_string_map[int(all_skeleton_id_string_key)] = all_skeleton_id_ts[all_skeleton_id_string_key]
#   all_skeleton_id_file.close()
#   type_content_data[all_skeleton_id] = all_skeleton_id_string_map
#   all_token_summary_ts[SkeletonNum] = len(all_skeleton_id_ts)
  
  print(MaximumStringLength + ":" + str(all_token_summary_ts[MaximumStringLength]))
  print(SkeletonNum + ":" + str(all_token_summary_ts[SkeletonNum]))
  print(SkeletonHitNum + ":" + str(all_token_summary_ts[SkeletonHitNum]))
  print(TokenNum + ":" + str(all_token_summary_ts[TokenNum]))
  print(TokenHitNum + ":" + str(all_token_summary_ts[TokenHitNum]))
  print(SwordNum + ":" + str(all_token_summary_ts[SwordNum]))
  print(SwordHitNum + ":" + str(all_token_summary_ts[SwordHitNum]))
  print(CharNum + ":" + str(all_token_summary_ts[CharNum]))
  print(CharHitNum + ":" + str(all_token_summary_ts[CharHitNum]))
#   print(TotalNumberOfSubWord + ":" + str(all_token_summary_ts[TotalNumberOfSubWord]))
#   print(TotalNumberOfChar + ":" + str(all_token_summary_ts[TotalNumberOfChar]))
  
  ''' in cascade char form '''
  all_token_subword_sequences_file = open(data_dir + "/All_token_subword_sequences.json", 'r', encoding='UTF-8')
  all_token_subword_sequences_ts = json.load(all_token_subword_sequences_file)
  all_token_subword_sequences_file.close()
  type_content_data[all_token_subword_sequences] = tf.convert_to_tensor(all_token_subword_sequences_ts)
  
  all_token_each_subword_sequence_start_file = open(data_dir + "/All_token_each_subword_sequence_start.json", 'r', encoding='UTF-8')
  all_token_each_subword_sequence_start_ts = json.load(all_token_each_subword_sequence_start_file)
  all_token_each_subword_sequence_start_file.close()
  type_content_data[all_token_each_subword_sequence_start] = tf.convert_to_tensor(all_token_each_subword_sequence_start_ts)
  
  all_token_each_subword_sequence_end_file = open(data_dir + "/All_token_each_subword_sequence_end.json", 'r', encoding='UTF-8')
  all_token_each_subword_sequence_end_ts = json.load(all_token_each_subword_sequence_end_file)
  all_token_each_subword_sequence_end_file.close()
  type_content_data[all_token_each_subword_sequence_end] = tf.convert_to_tensor(all_token_each_subword_sequence_end_ts)
  
  all_subword_is_start_end_file = open(data_dir + "/All_subword_is_start_end.json", 'r', encoding='UTF-8')
  all_subword_is_start_end_ts = json.load(all_token_each_subword_sequence_end_file)
  all_subword_is_start_end_file.close()
  type_content_data[all_subword_is_start_end] = tf.convert_to_tensor(all_subword_is_start_end_ts)
  
  all_token_char_sequences_file = open(data_dir + "/All_token_char_sequences.json", 'r', encoding='UTF-8')
  all_token_char_sequences_ts = json.load(all_token_char_sequences_file)
  all_token_char_sequences_file.close()
  type_content_data[all_token_char_sequences] = tf.convert_to_tensor(all_token_char_sequences_ts)
  
  all_token_each_char_sequence_start_file = open(data_dir + "/All_token_each_char_sequence_start.json", 'r', encoding='UTF-8')
  all_token_each_char_sequence_start_ts = json.load(all_token_each_char_sequence_start_file)
  all_token_each_char_sequence_start_file.close()
  type_content_data[all_token_each_char_sequence_start] = tf.convert_to_tensor(all_token_each_char_sequence_start_ts)
  
  all_token_each_char_sequence_end_file = open(data_dir + "/All_token_each_char_sequence_end.json", 'r', encoding='UTF-8')
  all_token_each_char_sequence_end_ts = json.load(all_token_each_char_sequence_end_file)
  all_token_each_char_sequence_end_file.close()
  type_content_data[all_token_each_char_sequence_end] = tf.convert_to_tensor(all_token_each_char_sequence_end_ts)
  
  
