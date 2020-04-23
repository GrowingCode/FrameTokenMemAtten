from utils.config_util import check_or_store_configs
from models.model_runner import ModelRunner
import tensorflow as tf
from models.tree.tree_model_runner import TreeModelRunner
from metas.hyper_settings import model_run_mode, tree_decode_mode


if __name__ == '__main__':
#   tf.debugging.set_log_device_placement(True)
  check_or_store_configs()
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
    if model_run_mode < tree_decode_mode:
      runner = ModelRunner(sess)
    elif model_run_mode == tree_decode_mode:
      runner = TreeModelRunner(sess)
    else:
      assert False, "unrecognized model_run_mode:" + str(model_run_mode)
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    runner.train()
    runner.test()
  
  


