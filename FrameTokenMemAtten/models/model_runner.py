import json
import os
import time

from inputs.type_content_data_loader import load_type_content_data
from metas.hyper_settings import top_ks, restrain_maximum_count, max_train_epoch, \
  valid_epoch_period, ignore_restrain_count, max_examples_in_one_batch, \
  gradient_clip_abs_range
from metas.non_hyper_constants import data_dir, model_storage_dir, turn_info, \
  turn, model_check_point, model_best, best, best_info, model_config, \
  np_float_type, testing_mode, training_mode, validating_mode
from models.skeleton_decoder import SkeletonDecodeModel
import numpy as np
import tensorflow as tf
from utils.config_util import check_or_store_configs


class ModelRunner():
  
  def __init__(self):
    self.optimizer = tf.keras.optimizers.Adam()
    '''
    load training data
    '''
    self.train_raw_datas = raw_load_data(data_dir + "/" + "train_data.txt", "train")
    self.train_np_arrays = []
    '''
    load valid data
    currently valid data is not considered
    '''
    self.valid_raw_datas = raw_load_data(data_dir + "/" + "valid_data.txt", "valid")
    self.valid_np_arrays = []
    '''
    load test data
    '''
    self.test_raw_datas = raw_load_data(data_dir + "/" + "test_data.txt", "test")
    self.test_np_arrays = []
    '''
    load type content data
    '''
    self.type_content_data = {}
    load_type_content_data(self.type_content_data)
    '''
    Initialize the directory to put the stored model
    '''
    if not os.path.exists(model_storage_dir):
      os.makedirs(model_storage_dir)
    self.turn_info_txt = model_storage_dir + '/' + turn_info
    self.turn_txt = model_storage_dir + '/' + turn
    self.check_point_model = model_storage_dir + '/' + model_check_point
    self.best_info_txt = model_storage_dir + '/' + best_info
    self.best_txt = model_storage_dir + '/' + best
    self.best_model = model_storage_dir + '/' + model_best
    self.config_txt = model_storage_dir + '/' + model_config
  
  def train(self):
    turn = 0
    min_loss = None
    max_accuracy = None
    restrain_count = 0
    turn_info = []
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
          turn_info.append(line)
    '''
    restore model when turn is not 0
    '''
    if turn == 0:
      model = SkeletonDecodeModel(self.type_content_data)
    else:
      if restrain_count >= restrain_maximum_count:
        turn = max_train_epoch
      if turn < max_train_epoch:
        model = tf.keras.models.load_model(self.check_point_model)
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
      train_output_result = self.model_running(model, training_mode)
      train_end_time = time.time()
      train_time_cost = train_end_time - train_start_time
      total_turn_sum_train_time_cost = total_turn_sum_train_time_cost + train_time_cost
      total_turn_sum = total_turn_sum + 1.0
      total_turn_average_train_time_cost = total_turn_sum_train_time_cost / total_turn_sum
      ''' exactly record train loss '''
      train_avg = compute_average(train_output_result)
      train_average_loss = train_avg["average_all_loss"]
      '''
      compute average loss when training
      '''
      print(str(turn+1) + "/" + str(max_train_epoch) + " turn's train_set_average_loss:" + str(train_average_loss))# + "#" + json.dumps(last_not_zero_average_accuracy));
      '''
      valid the training process if period reached
      '''
      train_compute_valid = (turn+1) % valid_epoch_period == 0
      if train_compute_valid:
        valid_output_result = self.model_running(model, validating_mode)
        valid_avg = compute_average(valid_output_result)
        '''
        compute average loss
        '''
        valid_average_loss = 0.0
        valid_average_accuracy = np.zeros([len(top_ks)], dtype=np_float_type).tolist()
        if valid_avg:
          valid_average_loss = valid_avg["average_all_loss"]
          valid_average_accuracy = valid_avg["average_all_accurate"]
          print(str(turn+1) + "/" + str(max_train_epoch) + " turn" + "#" + json.dumps(valid_avg));
        '''
        save best model
        '''
        if max_accuracy is None:
          max_accuracy = valid_average_accuracy
          min_loss = valid_average_loss
          print("========== Saving best model ==========")
          model.save(self.best_model)
          with open(self.best_info_txt, 'w') as best_info_record:
            best_info_record.write("the_turn_generating_best_model:" + str(turn+1) + "#" + dict_to_string(valid_avg))
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
            print("========== Saving best model ==========")
            model.save(self.best_model)
            with open(self.best_info_txt, 'w') as best_info_record:
              best_info_record.write("the_turn_generating_best_model:" + str(turn+1) + "#" + dict_to_string(valid_avg))
          else:
            restrain_count = restrain_count + 1
        turn_info.append(dict_to_string(valid_avg))
        '''
        save turn model
        judge whether the model is best currently
        the following if judgment is to decide whether the model has been restrained for a while, if so stop instantly
        save the model if period reached
        save check point model
        '''
        print("========== Saving check point model ==========")
        model.save(self.check_point_model)
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
          t_info_record = '\n'.join(turn_info)
          turn_info_record.write(t_info_record)
        '''
        go to next epoch
        '''
      if (not ignore_restrain_count) and (restrain_count >= restrain_maximum_count):
        turn = max_train_epoch
      turn = turn+1
  
  def test(self):
    print("===== Testing procedure starts! =====")
    print("Restore best model in " + self.best_model)
    model = tf.keras.models.load_model(self.best_model)
    '''
    compute average loss
    test set loss/accuracy leaves_score/all_score
    '''
    output_result = self.model_running(model, testing_mode)
    avg = compute_average(output_result)
    with open(self.best_txt, 'w') as best_model_statement_accuracy_record:
      best_model_statement_accuracy_record.write(json.dumps(avg))
    print(dict_to_string(avg))
  
  def model_running(self, model, mode):
    '''
    mode takes three values:
    0:train
    1:valid
    2:test
    '''
    all_metrics = {}
    if mode == 0:
      np_arrays = self.train_np_arrays
    elif mode == 1:
      np_arrays = self.valid_np_arrays
    else:
      np_arrays = self.test_np_arrays
    part_np_arrays = []
    one_unit_count = 0
    length_of_datas = len(np_arrays)
    for element_number, np_array in enumerate(np_arrays):
      part_np_arrays.append(np_array)
      if (one_unit_count >= max_examples_in_one_batch) or (element_number == length_of_datas-1):
        part_metric = self.model_running_batch(model, mode, part_np_arrays)
        merge_metric(all_metrics, part_metric)
        part_np_arrays.clear()
        one_unit_count = 0
    return all_metrics
  
  def model_running_batch(self, model, mode, part_np_arrays):
    final_result = {}
    '''
    put raw data into many batches, each batch contains many training examples
    each example has the following shape: (input_unit, input_unit, ...)
    for an example input_unit is ast_encodes_tensor or ast_decodes_tensor or sequence_decodes_tensor
    for each input_unit, the shape is [3 or 4, ast_node_num or sequence_num]
    3 or 4 is the first dimension of ast tensor or sequence tensor for one Java code
    '''
    for np_array in part_np_arrays:
      start_time = time.time()
      output_result = self.model_running_one_example(model, mode, np_array)
      merge_metric(final_result, output_result)
      end_time = time.time()
      avg = None
      if mode > training_mode:
        avg = compute_average(output_result)
    print("batch_size:" + str(len(part_np_arrays)) + "#time_cost:" + str(round(end_time-start_time, 1)) +"s" + "#" + json.dumps(avg))
    return final_result
  
  def model_running_one_example(self, model, mode, np_array):
    training = False
    if mode == training_mode:
      training = True
    if training:
      with tf.GradientTape() as tape:
        metrics = model(np_array[0], np_array[1], np_array[2], training = training)
        grads = tape.gradient(metrics[model.metrics_index["all_loss"]], model.trainable_variables)
        grads = clip_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    else:
      metrics = model(np_array[0], np_array[1], np_array[2], training = training)
    return metrics


def raw_load_data(data_file_name, mode_info):
  start_time = time.time()
  with open(data_file_name) as data_file:
    raw_datas = data_file.readlines()
  end_time = time.time()
  print("reading " + mode_info + " raw data using " + str(end_time-start_time) +"s")
  return raw_datas


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
      if all_metrics_is_empty:
        all_metrics[key] = p_one_item
      else:
        all_metrics[key] = all_metrics[key] + p_one_item


def compute_average(dict_t):
  r = {}
  for k in dict_t:
    if not k.endswith('_count'):
      idx = k.find('_')
      assert idx > 0
      first_k_part = k[0:idx]
      k_count = first_k_part + '_count'
      k_tm = "average_" + k
      divd = dict_t[k_count]
      if divd == 0.0:
        divd = 0.0000000001
      r[k_tm] = de_numpy(dict_t[k]/divd)
  return r


def de_numpy(d):
  if isinstance(d, (list, tuple, np.ndarray)):
    return d.tolist()
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
  
  
def clip_gradients(grads):
  final_grads = []
  for (gv, var) in grads:
    if gv is not None:
      grad = tf.clip_by_value(gv, -gradient_clip_abs_range, gradient_clip_abs_range)
      final_grads.append((grad, var))
  return final_grads
  

if __name__ == '__main__':
  check_or_store_configs()
  runner = ModelRunner()
  runner.train()
  runner.test()
  


