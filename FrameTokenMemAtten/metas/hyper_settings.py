top_ks = [1, 3, 6, 10]
mrr_max = top_ks[-1]
num_units = 128
contingent_parameters_num = 20
use_dup_model = 1
accumulated_token_max_length = 600
compute_token_memory = 1
''' attention algorithm '''
v_attention = 0
stand_attention = 1
attention_algorithm = stand_attention
''' decode mode '''
sword_decode = 0
token_decode = 1
atom_decode_mode = token_decode

ignore_restrain_count = 0
restrain_maximum_count = 3
max_train_epoch = 30
valid_epoch_period = 1

'''
actually each example in one batch is still trained one by one
just truncate the whole large data
'''
max_examples_in_one_batch = 128
''' gradient '''
gradient_clip_abs_range = 500.0

