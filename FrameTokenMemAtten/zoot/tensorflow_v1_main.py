from utils.config_util import check_or_store_configs
import tensorflow as tf
from models.model_runner import ModelRunner


if __name__ == '__main__':
#   tf.debugging.set_log_device_placement(True)
  check_or_store_configs()
  tf.compat.v1.disable_eager_execution()
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False)) as sess:
    runner = ModelRunner(sess)
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    runner.train()
    runner.test()
  
  


