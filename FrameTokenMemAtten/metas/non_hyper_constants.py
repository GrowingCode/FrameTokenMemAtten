import tensorflow as tf
import numpy as np
import os


home_dir = os.path.expanduser('~')
data_dir = home_dir + "/" + "AST_Tensors"

bool_type = tf.bool
float_type = tf.float32
int_type = tf.int32

bp_bool_type = np.bool
np_float_type = np.float32
np_int_type = np.int32

all_token_summary = "all_token_summary"
# all_skeleton_id = "all_skeleton_id"

SkeletonNum = "SkeletonNum"
SkeletonHitNum = "SkeletonHitNum"
TokenNum = "TokenNum"
TokenHitNum = "TokenHitNum"
SwordNum = "SwordNum"
SwordHitNum = "SwordHitNum"
CharNum = "CharNum"
CharHitNum = "CharHitNum"
# TotalNumberOfSubWord = "TotalNumberOfSubWord"
# TotalNumberOfChar = "TotalNumberOfChar"
MaximumStringLength = "MaximumStringLength"

all_token_subword_sequences = "all_token_subword_sequences"
all_token_each_subword_sequence_start = "all_token_each_subword_sequence_start"
all_token_each_subword_sequence_end = "all_token_each_subword_sequence_end"
all_subword_is_start_end = "all_subword_is_start_end"

all_token_char_sequences = "all_token_char_sequences"
all_token_each_char_sequence_start = "all_token_each_char_sequence_start"
all_token_each_char_sequence_end = "all_token_each_char_sequence_end"

model_storage_parent_dir = 'zoot'
model_storage_dir = "zoot_run_info_record"
model_config = "model_config.txt"
model_check_point = "model_check_point"
model_best = "model_best"
turn_info = "turn_info.txt"
turn = "turn.txt"
best_info = "best_info.txt"
best = "best.txt"

'''
model_running_mode
'''
training_mode = 0
validating_mode = 1
testing_mode = 2

'''
length threshold for an example
'''
max_threshold_example_length = 2000
'''
default ID
'''
UNK_en = 1









