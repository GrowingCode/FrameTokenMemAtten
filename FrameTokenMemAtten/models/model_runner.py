import json
import os
import time
from inputs.type_content_data_loader import load_type_content_data
from metas.hyper_settings import top_ks, restrain_maximum_count, max_train_epoch, \
  valid_epoch_period, ignore_restrain_count, max_examples_in_one_batch, \
  gradient_clip_abs_range
from metas.non_hyper_constants import model_storage_dir, turn_info, \
  turn, model_check_point, model_best, best, best_info, model_config, \
  np_float_type, testing_mode, training_mode, validating_mode,\
  model_storage_parent_dir, test_noavg, validate_noavg, train_noavg
import numpy as np
import tensorflow as tf


class ModelRunner():
  
  def __init__(self, sess):
    '''
    load training data
    '''
    self.set_up_example_loader()
#     data_dir + "/" + "tree_train_data.txt", 
    self.train_np_arrays = load_examples("train", self.example_loader)
    '''
    load valid data
    currently valid data is not considered
    '''
#     data_dir + "/" + "tree_valid_data.txt", 
    self.valid_np_arrays = load_examples("valid", self.example_loader)
    '''
    load test data
    '''
#     data_dir + "/" + "tree_test_data.txt", 
    self.test_np_arrays = load_examples("test", self.example_loader)
    '''
    load type content data
    '''
    self.type_content_data = {}
    load_type_content_data(self.type_content_data)
    '''
    initialize the directory to put the stored model
    '''
    real_model_storage_dir = '../' + model_storage_parent_dir + '/' + model_storage_dir
    if not os.path.exists(real_model_storage_dir):
      os.makedirs(real_model_storage_dir)
    self.turn_info_txt = real_model_storage_dir + '/' + turn_info
    self.turn_txt = real_model_storage_dir + '/' + turn
    self.check_point_directory = real_model_storage_dir + '/' + model_check_point
    self.check_point_file = self.check_point_directory + '/' + 'model.ckpt'
    self.best_info_txt = real_model_storage_dir + '/' + best_info
    self.best_txt = real_model_storage_dir + '/' + best
    self.best_model_directory = real_model_storage_dir + '/' + model_best
    self.best_model_file = self.best_model_directory + '/' + 'model.ckpt'
    self.config_txt = real_model_storage_dir + '/' + model_config
    '''
    files to store each token accuracy or each token atom accuracy data
    '''
    self.train_noavg_txt = real_model_storage_dir + '/' + train_noavg
    self.test_noavg_txt = real_model_storage_dir + '/' + test_noavg
    self.validate_noavg_txt = real_model_storage_dir + '/' + validate_noavg
    '''
    set up necessary data
    '''
    self.sess = sess
    place_holders = self.build_input_place_holder()
    self.build_model_logic()
    self.optimizer = tf.compat.v1.train.AdamOptimizer()
    '''
    build graph of logic 
    '''
    self.test_metrics = self.model(place_holders, training = False)
    assert isinstance(self.test_metrics, list)
#     self.test_metrics[-1] = convert_tensor_array_to_lists_of_tensors(make_sure_shape_of_tensor_array(self.test_metrics[-1]))
#     with tf.device('/GPU:0'):
    self.train_metrics = self.model(place_holders, training = True)
#     self.train_metrics[-1] = tf.constant(0, int_type)
    gvs = self.optimizer.compute_gradients(self.train_metrics[self.model.metrics_index["all_loss"]], tf.compat.v1.trainable_variables(), colocate_gradients_with_ops=True)
    final_grads = []
    for (gv, var) in gvs:
      if gv is not None:
        grad = tf.clip_by_value(gv, -gradient_clip_abs_range, gradient_clip_abs_range)
        final_grads.append((grad, var))
    self.train_op = self.optimizer.apply_gradients(final_grads)
  
  def set_up_example_loader(self):
    pass
  
  def build_input_place_holder(self):
    pass
  
  def build_model_logic(self):
    pass
  
  def build_feed_dict(self, one_example):
    pass
  
  def train(self):
    turn = 0
    min_loss = None
    max_accuracy = None
    restrain_count = 0
    turn_infos = []
    if os.path.exists(self.turn_txt):
      assert os.path.exists(self.turn_info_txt)
      with open(self.turn_txt, 'r') as turn_record:
        turn_record_json = json.load(turn_record)
        turn = turn_record_json["turn"]
        min_loss = None if "min_loss" not in turn_record_json is None else turn_record_json["min_loss"]
        max_accuracy = None if "max_accuracy" not in turn_record_json is None else turn_record_json["max_accuracy"]
        restrain_count = turn_record_json["restrain_count"]
      with open(self.turn_info_txt, 'r') as turn_info_record:
        turn_info_lines = turn_info_record.readlines()
        for line in turn_info_lines:
          turn_infos.append(line)
    '''
    restore model when turn is not 0
    '''
    
    if restrain_count >= restrain_maximum_count:
      turn = max_train_epoch
    if turn > 0 and turn < max_train_epoch:
      tf.compat.v1.train.Saver().restore(self.sess, self.check_point_file)
    '''
    begin real training procedure
    '''
    total_turn_sum_train_time_cost = 0.0
    total_turn_sum = 0.0
    while turn < max_train_epoch:
      '''
      one epoch starts
      '''
      train_start_time = time.time()
      train_output_result = self.model_running(training_mode)
      train_end_time = time.time()
      train_time_cost = train_end_time - train_start_time
      total_turn_sum_train_time_cost = total_turn_sum_train_time_cost + train_time_cost
      total_turn_sum = total_turn_sum + 1.0
      total_turn_average_train_time_cost = total_turn_sum_train_time_cost / total_turn_sum
      ''' exactly record train loss '''
      train_avg = compute_average(train_output_result)
      train_noavg = process_noavg(train_output_result)
      with open(self.train_noavg_txt, 'w') as train_noavg_record:
        train_noavg_record.write(dict_to_string(train_noavg))
      train_average_loss = train_avg["average_all_loss"]
      '''
      compute average loss when training
      '''
      print(str(turn+1) + "/" + str(max_train_epoch) + " turn's train_set_average_loss:" + str(train_average_loss))
      '''
      valid the training process if period reached
      '''
      train_compute_valid = (turn+1) % valid_epoch_period == 0
      if train_compute_valid:
        valid_output_result = self.model_running(validating_mode)
        valid_avg = compute_average(valid_output_result)
        valid_noavg = process_noavg(valid_output_result)
        '''
        compute average loss
        '''
        valid_average_loss = 0.0
        valid_average_accuracy = np.zeros([len(top_ks)], dtype=np_float_type).tolist()
        if valid_avg:
          valid_average_loss = valid_avg["average_all_loss"]
          valid_average_accuracy = valid_avg["average_all_accurate"]
          print(str(turn+1) + "/" + str(max_train_epoch) + " turn" + "#" + json.dumps(valid_avg))
        '''
        save best model
        '''
        to_save_best_model = False
        if max_accuracy is None:
          max_accuracy = valid_average_accuracy
          min_loss = valid_average_loss
          to_save_best_model = True
        else:
          if newer_accuracy_is_better(max_accuracy, valid_average_accuracy):
            print("max_accuracy[0]:" + str(max_accuracy[0]) + "#valid_average_accuracy[0]:" + str(valid_average_accuracy[0]))
            if round(max_accuracy[0], 6) == round(valid_average_accuracy[0], 6):
              restrain_count = restrain_count + 1
              print("restrain_count:" + str(restrain_count))
            else:
              restrain_count = 0
            max_accuracy = valid_average_accuracy
            min_loss = valid_average_loss
            to_save_best_model = True
          else:
            restrain_count = restrain_count + 1
        if to_save_best_model:
          print("========== Saving best model ==========")
          tf.compat.v1.train.Saver().save(self.sess, self.best_model_file)
          with open(self.best_info_txt, 'w') as best_info_record:
            best_info_record.write("the_turn_generating_best_model:" + str(turn+1) + "#" + dict_to_string(valid_avg))
          with open(self.validate_noavg_txt, 'w') as validate_noavg_record:
            validate_noavg_record.write(dict_to_string(valid_noavg))
          
        turn_infos.append(dict_to_string(valid_avg))
        '''
        save turn model
        judge whether the model is best currently
        the following if judgment is to decide whether the model has been restrained for a while, if so stop instantly
        save the model if period reached
        save check point model
        '''
        print("========== Saving check point model ==========")
        tf.compat.v1.train.Saver().save(self.sess, self.check_point_file)
        '''
        write the turn to file
        '''
        turn_record_json2 = {}
        with open(self.turn_txt, 'w') as turn_record:
          turn_record_json2["turn"] = turn+1
          turn_record_json2["restrain_count"] = restrain_count
          turn_record_json2["min_loss"] = min_loss
          turn_record_json2["max_accuracy"] = max_accuracy
          turn_record_json2["average_train_time_cost"] = total_turn_average_train_time_cost
          turn_record.write(json.dumps(turn_record_json2))
        with open(self.turn_info_txt, 'w') as turn_info_record:
          t_info_record = '\n'.join(turn_infos)
          turn_info_record.write(t_info_record)
        '''
        go to next epoch
        '''
      if (not ignore_restrain_count) and (restrain_count >= restrain_maximum_count):
        turn = max_train_epoch
      turn = turn+1
  
  def test(self):
    print("===== Testing procedure starts! =====")
    print("Restore best model in " + self.best_model_directory)
    tf.compat.v1.train.Saver().restore(self.sess, self.best_model_file)
    '''
    compute average loss
    test set loss/accuracy leaves_score/all_score
    '''
    output_result = self.model_running(testing_mode)
    avg = compute_average(output_result)
    noavg = process_noavg(output_result)
    with open(self.best_txt, 'w') as best_model_statement_accuracy_record:
      best_model_statement_accuracy_record.write(json.dumps(avg))
    with open(self.test_noavg_txt, 'w') as test_noavg_record:
      test_noavg_record.write(json.dumps(noavg))
    print(dict_to_string(avg))
  
  def model_running(self, mode):
    '''
    mode takes three values:
    0:train
    1:valid
    2:test
    '''
    training = False
    if mode == training_mode:
      training = True
    '''
    feed data to run model
    '''
    all_metrics = {}
    if mode == 0:
      mode_str = "train"
      np_arrays = self.train_np_arrays
    elif mode == 1:
      mode_str = "validate"
      np_arrays = self.valid_np_arrays
    else:
      mode_str = "test"
      np_arrays = self.test_np_arrays
    part_np_arrays = []
    one_unit_count = 0
    length_of_datas = len(np_arrays)
    for element_number, np_array in enumerate(np_arrays):
      one_unit_count = one_unit_count + 1
      part_np_arrays.append(np_array)
      if (one_unit_count >= max_examples_in_one_batch) or (element_number == length_of_datas-1):
        start_time = time.time()
        for np_array in part_np_arrays:
          part_metric = self.model_running_one_example(training, np_array)
          part_metric = model_output(part_metric, self.model.statistical_metrics_meta)
          merge_metric(all_metrics, part_metric)
        batch_size = len(part_np_arrays)
        assert batch_size == one_unit_count
        part_np_arrays.clear()
        one_unit_count = 0
        end_time = time.time()
        print("mode:" + mode_str + "#batch_size:" + str(batch_size) + "#time_cost:" + str(round(end_time-start_time, 1)) +"s")
    return all_metrics
  
  def model_running_one_example(self, training, one_example):
    feed_dict = self.build_feed_dict(one_example)
    if training:
      r_metrics = self.sess.run([self.train_op, *self.train_metrics], feed_dict=feed_dict)
      metrics = r_metrics[1:]
    else:
      metrics = self.sess.run([*self.test_metrics], feed_dict=feed_dict)
    return metrics


def load_examples(mode_info, example_loader):
  start_time = time.time()
  end_time = time.time()
  print("reading " + mode_info + " raw data using " + str(end_time-start_time) +"s")
  start_time = time.time()
  examples = example_loader(mode_info)
  end_time = time.time()
  print("pre_processing " + mode_info + " raw data using " + str(end_time-start_time) +"s")
  return examples


def merge_metric(all_metrics, part_metric):
  ''' assert two structures are same '''
  all_metrics_is_empty = True
  if all_metrics:
    all_metrics_is_empty = False
  for key in part_metric:
    if not all_metrics_is_empty:
      assert key in all_metrics
    p_one_item = part_metric[key]
    if type(p_one_item) == dict:
      if all_metrics_is_empty:
        all_metrics[key] = {}
      merge_metric(all_metrics[key], p_one_item)
    else:
      if key.endswith("_noavg"):
        if all_metrics_is_empty:
          all_metrics[key] = []
        all_metrics[key].append(p_one_item)
      else:
        if all_metrics_is_empty:
          all_metrics[key] = p_one_item
        else:
          all_metrics[key] = all_metrics[key] + p_one_item


def compute_average(dict_t):
  r = {}
  for k in dict_t:
    if not k.endswith('_count') and not k.endswith('_noavg'):
      idx = k.find('_')
      assert idx > 0
      first_k_part = k[0:idx]
      k_count = first_k_part + '_count'
      k_tm = "average_" + k
      divd = dict_t[k_count]
      if divd == 0.0:
        divd = 0.0000000001
      r[k_tm] = de_numpy(dict_t[k]/divd)
    elif k.endswith('_count'):
      r[k] = de_numpy(dict_t[k])
  return r


def process_noavg(dict_t):
  r = {}
  for k in dict_t:
    if k.endswith('_noavg'):
      r[k] = de_numpy(dict_t[k])
  pass


def de_numpy(d):
  if isinstance(d, (np.ndarray)):
    return d.tolist()
  elif isinstance(d, (list, tuple)):
    r = []
    for ele in d:
      r.append(de_numpy(ele))
    return r;
  else:
    if isinstance(d, (np.float16, np.float32, np.float64)):
      return float(d)
    if isinstance(d, (np.int8, np.int16, np.int32, np.int64)):
      return int(d)


def dict_to_string(dict_t):
  r = ""
  for k in dict_t:
    t = dict_t[k]
    if (type(t) == dict):
      r = r + dict_to_string(t)
    else:
      r = r + "#" + k + ":" + str(t)
  return r


def newer_accuracy_is_better(old_accuracy, new_accuracy):
  for i in range(len(top_ks)):
    if old_accuracy[i] < new_accuracy[i]:
      return True
    elif old_accuracy[i] > new_accuracy[i]:
      return False


def info_of_train_stop_test_start(average_accuracy):
  print("===== Because training accuracy is very high:" + str(average_accuracy) + ", training procedure will stop soon! =====")
  time.sleep(1)
  

def model_output(tensors, tensors_meta):
  assert len(tensors) == len(tensors_meta), "tensors length:" + str(len(tensors)) + "#tensors_meta length:" + str(len(tensors_meta))
  numpy_dict = {}
  for i, t in enumerate(tensors):
    numpy_dict[tensors_meta[i][0]] = t
  return numpy_dict
  
  
def clip_gradients(grads, vs):
  final_grads = []
  for (gv, v) in zip(grads, vs):
    if gv is not None:
      grad = tf.clip_by_value(gv, -gradient_clip_abs_range, gradient_clip_abs_range)
      final_grads.append((grad, v))
  return final_grads



