'''
initialize settings
'''
initialize_range = 1.0
lstm_initialize_range = 1.0
initialize_seed_base = 0
'''
basic learning mode
'''
token_in_scope_valid = 0
token_meaningful_valid = 1
token_valid_mode = token_meaningful_valid
consider_all_token_accuracy = 0
only_consider_var_accuracy = 1
only_consider_unseen_var_accuracy = 2
token_accuracy_mode = consider_all_token_accuracy
'''
details
'''
sequence_decode_mode = 0
statement_decode_mode = 1
skeleton_decode_mode = 2
tree_decode_mode = 3
model_run_mode = skeleton_decode_mode
''' tree decode details '''
tree_decode_with_grammar = 0
tree_leaf_one_more_lstm_step = 0
tree_decode_2d = 0
tree_decode_embed = 1
tree_decode_way = tree_decode_2d
''' statistics '''
top_ks = [1, 3, 6, 10]
mrr_max = top_ks[-1]
num_units = 128
contingent_parameters_num = 20
accumulated_token_max_length = 600
'''
basic lstm mode
'''
use_layer_norm = 1
use_lstm_merger_style = 0
''' memory mode '''
no_memory_mode = 0
only_memory_mode = 1
concat_memory_mode = 2
abs_size_concat_memory_mode = 3
abs_size_var_novar_all_concat_memory_mode = 4
token_memory_mode = no_memory_mode
abs_memory_size = 25
''' take unseen as UNK '''
take_unseen_as_UNK = 1
''' token_embedder_mode '''
token_only_mode = 0
swords_compose_mode = 1
token_embedder_mode = token_only_mode
''' dup_mode '''
use_dup_model = 0
use_syntax_to_decide_rep = 0
''' dup share parameters '''
dup_share_parameters = 0
''' is_dup_mode '''
max_repetition_mode = 0
attention_repetition_mode = 1
repetition_mode = attention_repetition_mode
''' is_dup_algorithm '''
simple_is_dup = 0
mlp_is_dup = 1
sigmoid_is_dup = 2
is_dup_mode = sigmoid_is_dup
dup_use_two_poles = 0
''' dup accuracy mode '''
en_match = 0
exact_accurate = 1
repetition_accuracy_mode = en_match
''' attention algorithm '''
v_attention = 0
stand_attention = 1
attention_algorithm = stand_attention
''' decode mode '''
token_decode = 0
sword_decode = 1
atom_decode_mode = token_decode
''' training hyper '''
ignore_restrain_count = 0
restrain_maximum_count = 2
max_train_epoch = 100
valid_epoch_period = 1
'''
actually each example in one batch is still trained one by one
just truncate the whole large data
'''
max_examples_in_one_batch = 128
''' gradient '''
gradient_clip_abs_range = 1.0
'''
whether compose tokens
'''
compute_token_memory = 0
compose_tokens_of_a_statement = 0
compose_one_way_lstm = 0
compose_bi_way_lstm = 1
compose_mode = compose_one_way_lstm
'''
decode attention, high level attention way
'''
decode_no_attention = 0
decode_with_attention = 1
decode_attention_way = decode_no_attention


'''
additional composite configuration
'''
composite_config_func = "skeleton_token_decode"

if composite_config_func == "skeleton_token_decode":
  model_run_mode = skeleton_decode_mode

if composite_config_func == "skeleton_token_decode_with_dup":
  model_run_mode = skeleton_decode_mode
  use_dup_model = 1
  compute_token_memory = 1
  token_memory_mode = only_memory_mode
  
if composite_config_func == "statement_token_decode":
  model_run_mode = statement_decode_mode
  
if composite_config_func == "statement_token_decode_with_tokens_compose":
  model_run_mode = statement_decode_mode
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1
  
if composite_config_func == "statement_token_decode_with_bi_way_tokens_compose":
  model_run_mode = statement_decode_mode
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1
  compose_mode = compose_bi_way_lstm
  
if composite_config_func == "statement_token_decode_with_dup":
  model_run_mode = statement_decode_mode
  use_dup_model = 1
  compute_token_memory = 1
  token_memory_mode = only_memory_mode
  
if composite_config_func == "sequence_token_decode":
  model_run_mode = sequence_decode_mode
  
if composite_config_func == "sequence_token_decode_with_dup":
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = concat_memory_mode
  
if composite_config_func == "sequence_token_decode_with_rep_dup":
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = concat_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "sequence_token_decode_with_abs_concat_dup":
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = abs_direct_concat_memory_mode
  
if composite_config_func == "tree_token_decode":
  model_run_mode = tree_decode_mode
  

'''
configuration hard checking
'''
if atom_decode_mode == sword_decode:
  assert token_embedder_mode == swords_compose_mode
if compose_tokens_of_a_statement == 1:
  assert compute_token_memory == 1
if dup_use_two_poles:
  assert repetition_mode == max_repetition_mode
if use_dup_model:
  assert token_memory_mode > no_memory_mode
if compose_tokens_of_a_statement:
  assert model_run_mode == statement_decode_mode
  
  
  
  
  






