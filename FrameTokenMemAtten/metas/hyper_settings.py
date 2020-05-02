from inputs.example_data_loader import build_skeleton_feed_dict,\
  build_statement_feed_dict, build_sequence_feed_dict, build_tree_feed_dict


'''
basic learning mode
'''
token_in_scope_valid = 0
token_meaningful_valid = 1
token_valid_mode = token_meaningful_valid
only_consider_var_accuracy = 0
'''
details
'''
build_feed_dict = build_skeleton_feed_dict
sequence_decode_mode = 0
skeleton_decode_mode = 1
tree_decode_mode = 2
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
use_tensorflow_lstm_form = 0
use_lstm_merger_style = 0
''' memory mode '''
no_memory_mode = 0
only_memory_mode = 1
concat_memory_mode = 2
# bilstm_memory_mode = 3
# bilstm_memory_concat_mode = 4
token_memory_mode = no_memory_mode
''' take unseen as UNK '''
take_unseen_as_UNK = 1
''' whether skeleton '''
treat_first_element_as_skeleton = 1
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
gradient_clip_abs_range = 500.0
'''
whether compose tokens
'''
compose_tokens_of_a_statement = 0
stand_compose = 0
compose_for_attention_use = 1
two_way_compose = 2
three_way_compose = 3
compose_mode = stand_compose
compose_share_parameters = 1
'''
decode attention, high level attention way
'''
decode_no_attention = 0
decode_with_attention = 1
decode_attention_way = decode_no_attention


'''
additional composite configuration
'''
composite_config_func = "only_token_decode"

if composite_config_func == "only_token_decode":
  pass

if composite_config_func == "only_token_decode_with_tokens_compose":
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1

if composite_config_func == "only_token_decode_with_dup":
  use_dup_model = 1
  
if composite_config_func == "only_token_decode_with_dup_with_two_poles":
  use_dup_model = 1
  dup_use_two_poles = 1
  repetition_mode = max_repetition_mode
  is_dup_mode = simple_is_dup
  
if composite_config_func == "only_token_decode_with_dup_share_dup_parameters":
  use_dup_model = 1
  dup_share_parameters = 1
  
if composite_config_func == "only_token_decode_with_dup_v_atten":
  use_dup_model = 1
  attention_algorithm = v_attention
  
if composite_config_func == "only_token_decode_with_memory_dup":
  use_dup_model = 1
  compute_token_memory = 1
  
if composite_config_func == "only_token_decode_with_memory_dup_v_atten":
  use_dup_model = 1
  compute_token_memory = 1
  attention_algorithm = v_attention
  
if composite_config_func == "only_token_decode_with_memory_dup_v_atten_with_lstm_states":
  use_dup_model = 1
  compute_token_memory = 1
  attention_algorithm = v_attention
  take_lstm_states_as_memory_states = 1
  
if composite_config_func == "token_decode_with_swords_comp_embed":
  token_embedder_mode = swords_compose_mode

if composite_config_func == "token_decode_with_swords_comp_embed_with_dup":
  use_dup_model = 1
  token_embedder_mode = swords_compose_mode
  
if composite_config_func == "token_decode_with_swords_comp_embed_with_tokens_compose":
  token_embedder_mode = swords_compose_mode
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1
  
if composite_config_func == "token_decode_with_swords_comp_embed_with_memory_dup":
  use_dup_model = 1
  compute_token_memory = 1
  token_embedder_mode = swords_compose_mode

if composite_config_func == "only_sword_decode":
  use_dup_model = 0
  atom_decode_mode = sword_decode
  token_embedder_mode = swords_compose_mode

if composite_config_func == "only_sword_decode_with_tokens_compose":
  use_dup_model = 0
  atom_decode_mode = sword_decode
  token_embedder_mode = swords_compose_mode
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1

if composite_config_func == "not_skeleton_only_token_decode":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  
if composite_config_func == "not_skeleton_only_token_decode_no_layer_norm":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  use_layer_norm = 0
  
if composite_config_func == "not_skeleton_only_token_decode_with_tensorflow_lstm":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  use_tensorflow_lstm_form = 1
  
if composite_config_func == "not_skeleton_only_token_decode_with_dup":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  use_dup_model = 1
  
if composite_config_func == "not_skeleton_only_token_decode_with_rep_dup":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  use_dup_model = 1
  is_dup_mode = simple_is_dup
  repetition_mode = max_repetition_mode
  
if composite_config_func == "not_skeleton_only_token_decode_with_dup_share_dup_parameters":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  use_dup_model = 1
  dup_share_parameters = 1
  
if composite_config_func == "not_skeleton_only_token_decode_with_tokens_compose":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1
  
if composite_config_func == "not_skeleton_only_token_decode_with_tokens_compose_no_layer_norm":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1
  use_layer_norm = 0
  
if composite_config_func == "not_skeleton_only_token_decode_with_tokens_compose_with_tensorflow_lstm":
  build_feed_dict = build_statement_feed_dict
  treat_first_element_as_skeleton = 0
  compute_token_memory = 1
  compose_tokens_of_a_statement = 1
  use_tensorflow_lstm_form = 1
  
if composite_config_func == "sequence_only_token_decode":
  build_feed_dict = build_sequence_feed_dict
  model_run_mode = sequence_decode_mode
  
if composite_config_func == "sequence_only_token_decode_with_dup":
  build_feed_dict = build_sequence_feed_dict
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = concat_memory_mode
  
if composite_config_func == "sequence_only_token_decode_with_syntax_dup":
  build_feed_dict = build_sequence_feed_dict
  model_run_mode = sequence_decode_mode
  use_dup_model = 1
  token_memory_mode = concat_memory_mode
  use_syntax_to_decide_rep = 1
  
if composite_config_func == "tree_token_decode":
  build_feed_dict = build_tree_feed_dict
  model_run_mode = tree_decode_mode
  

'''
configuration hard checking
'''
if atom_decode_mode == sword_decode:
  assert token_embedder_mode == swords_compose_mode
if treat_first_element_as_skeleton == 0:
  assert build_feed_dict == build_statement_feed_dict
if compose_tokens_of_a_statement == 1:
  assert compute_token_memory == 1
if dup_use_two_poles:
  assert repetition_mode == max_repetition_mode
if use_dup_model:
  assert token_memory_mode > no_memory_mode
    






