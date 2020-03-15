from utils.config_util import check_or_store_configs
import tensorflow as tf
from models.model_runner import ModelRunner


if __name__ == '__main__':
#   tf.debugging.set_log_device_placement(True)
  check_or_store_configs()
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    runner = ModelRunner(sess)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    runner.train()
    runner.test()
  
  


