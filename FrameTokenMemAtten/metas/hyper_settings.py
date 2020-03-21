from inputs.example_data_loader import build_skeleton_feed_dict,\
  build_statement_feed_dict


build_feed_dict = build_skeleton_feed_dict
''' statistics '''
top_ks = [1, 3, 6, 10]
mrr_max = top_ks[-1]
num_units = 128
contingent_parameters_num = 20
use_dup_model = 0
accumulated_token_max_length = 600
compute_token_memory = 0
''' whether skeleton '''
treat_first_element_as_skeleton = 1
''' token_embedder_mode '''
token_only_mode = 0
swords_compose_mode = 1
token_embedder_mode = token_only_mode
''' dup_mode '''
max_repetition_mode = 0
attention_repetition_mode = 1
repetition_mode = max_repetition_mode
''' is_dup_mode '''
simple_is_dup = 0
mlp_is_dup = 1
sigmoid_is_dup = 2
is_dup_mode=simple_is_dup
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
'''
whether compose tokens
'''
compose_tokens_of_a_statement = 0


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
  
if composite_config_func == "only_token_decode_with_memory_dup":
  use_dup_model = 1
  compute_token_memory = 1
  
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
  treat_first_element_as_skeleton = 0
  build_feed_dict = build_statement_feed_dict

'''
configuration hard checking
'''
if atom_decode_mode == sword_decode:
  assert token_embedder_mode == swords_compose_mode
if treat_first_element_as_skeleton == 0:
  assert build_feed_dict == build_statement_feed_dict








