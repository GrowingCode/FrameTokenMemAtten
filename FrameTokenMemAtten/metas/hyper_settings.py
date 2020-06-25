'''
initialize settings
'''
initialize_range = 1.0
lstm_initialize_range = 0.0
initialize_seed_base = 0
'''
compute extra valid each_noavg
'''
compute_extra_valid_each_noavg = 0
'''
basic learning mode
'''
token_in_scope_valid = 0
token_meaningful_valid = 1
token_valid_mode = token_meaningful_valid
consider_all_token_accuracy = 0
only_consider_token_kind_accuracy = 1
only_consider_var_accuracy = 2
only_consider_unseen_var_accuracy = 3
only_consider_non_var_accuracy = 4
token_accuracy_mode = consider_all_token_accuracy
ignore_unk_when_computing_accuracy = 0
''' token kind '''
default_token_kind = 0b0
simpletype_simplename = 0b01
qualified_simplename = 0b010
methodname_simplename = 0b0100
simplename_approximate_not_variable = 0b01000
simplename_approximate_variable = 0b010000
simplename = 0b0100000
non_leaf_at_least_two_children_without_qualified_node = 0b01000000
''' note that this directly influences the dup_pattern range in consideration '''
token_kind_consider_range_mode = simplename_approximate_variable
'''
details
'''
sequence_decode_mode = 0
skeleton_decode_mode = 1
statement_decode_mode = 2
tree_decode_mode = 3
linear_dup_mode = 4
skeleton_dup_mode = 5
statement_dup_mode = 6
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
use_layer_norm = 0
embed_merger_use_layer_norm = 0
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
''' dup base '''
dup_base_model_directory = "C:/Users/yangy/Desktop/Experiment/log4j/1000unk-sequence_token_decode/zoot_run_info_record"
''' dup classify mode '''
dup_all_classify = 0
dup_var_classify = 1
dup_in_token_kind_range_classify = 2
dup_classify_mode = dup_var_classify
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
restrain_maximum_count = 10
max_train_epoch = 200
valid_epoch_period = 1
'''
actually each example in one batch is still trained one by one
just truncate the whole large data
'''
max_examples_in_one_batch = 128
''' gradient '''
gradient_clip_abs_range = 1000000.0
'''
whether compose tokens
'''
compute_token_memory = 0
compose_tokens_of_a_statement = 0
compose_half_one_way_lstm = 0
compose_one_way_lstm = 1
compose_bi_way_lstm = 2
compose_mode = compose_one_way_lstm
one_way_stand_compose = 0
one_way_two_way_compose = 1
one_way_three_way_compose = 2
compose_one_way_lstm_mode = one_way_stand_compose
'''
decode attention, high level attention way
'''
decode_no_attention = 0
decode_with_attention = 1
decode_attention_way = decode_no_attention
'''
info print
'''
print_accurate_of_each_example = 0


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
  
if composite_config_func == "statement_token_decode_with_rep_dup":
  model_run_mode = statement_decode_mode
  use_dup_model = 1
  compute_token_memory = 1
  token_memory_mode = only_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "statement_token_decode_with_dup_with_sequence_concat_style":
  model_run_mode = statement_decode_mode
  use_dup_model = 1
  token_memory_mode = concat_memory_mode
  
if composite_config_func == "statement_token_decode_with_rep_dup_with_sequence_concat_style":
  model_run_mode = statement_decode_mode
  use_dup_model = 1
  token_memory_mode = concat_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "statement_token_decode_with_dup_with_sequence_only_memory_style":
  model_run_mode = statement_decode_mode
  use_dup_model = 1
  token_memory_mode = only_memory_mode
  
if composite_config_func == "statement_token_decode_with_rep_dup_with_sequence_only_memory_style":
  model_run_mode = statement_decode_mode
  use_dup_model = 1
  token_memory_mode = only_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
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
  
if composite_config_func == "sequence_token_decode_with_only_memory_rep_dup":
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = only_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "sequence_token_decode_with_abs_concat_rep_dup":
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = abs_size_concat_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "sequence_token_decode_with_abs_var_novar_concat_dup":
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = abs_size_var_novar_all_concat_memory_mode
  
if composite_config_func == "tree_token_decode":
  model_run_mode = tree_decode_mode
  
if composite_config_func == "tree_token_decode_with_tree_grammar":
  model_run_mode = tree_decode_mode
  tree_decode_with_grammar = 1
  
if composite_config_func == "linear_dup":
  model_run_mode = linear_dup_mode
  token_memory_mode = concat_memory_mode
  
if composite_config_func == "linear_rep_dup":
  model_run_mode = linear_dup_mode
  token_memory_mode = concat_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "linear_dup_only_memory_style":
  model_run_mode = linear_dup_mode
  token_memory_mode = only_memory_mode
  
if composite_config_func == "linear_rep_dup_only_memory_style":
  model_run_mode = linear_dup_mode
  token_memory_mode = only_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode

if composite_config_func == "statement_dup":
  model_run_mode = statement_dup_mode
  compute_token_memory = 1
  token_memory_mode = only_memory_mode
  
if composite_config_func == "statement_rep_dup":
  model_run_mode = statement_dup_mode
  compute_token_memory = 1
  token_memory_mode = only_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "statement_dup_sequence_style":
  model_run_mode = statement_dup_mode
  token_memory_mode = only_memory_mode
  
if composite_config_func == "statement_rep_dup_sequence_style":
  model_run_mode = statement_dup_mode
  token_memory_mode = only_memory_mode
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode

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
  
  
  
  
  






