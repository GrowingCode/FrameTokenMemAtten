from utils.config_util import check_or_store_configs
from models.model_runner import ModelRunner
import tensorflow as tf
from metas.non_hyper_constants import int_type


class ModelRunnerV1(ModelRunner):
  
  def __init__(self, sess):
    super(ModelRunnerV1, self).__init__()
    self.sess = sess
    self.skeleton_token_info_tensor = tf.compat.v1.placeholder(int_type, [3, None])
    self.skeleton_token_info_start_tensor = tf.compat.v1.placeholder(int_type, [None])
    self.skeleton_token_info_end_tensor = tf.compat.v1.placeholder(int_type, [None])
  
  def model_running_one_example(self, training, model, optimizer, token_info_tensor, token_info_start_tensor, token_info_end_tensor):
    with tf.device('/GPU:0'):
      metrics = model(token_info_tensor, token_info_start_tensor, token_info_end_tensor, training = training)
      gvs = optimizer.compute_gradients(metrics[model.metrics_index["all_loss"]], tf.compat.v1.trainable_variables(), colocate_gradients_with_ops=True)
      train = optimizer.apply_gradients(gvs)
    feed_dict = {self.skeleton_token_info_tensor:token_info_tensor, self.skeleton_token_info_start_tensor:token_info_start_tensor, self.skeleton_token_info_end_tensor:token_info_end_tensor}
    r_metrics = self.sess.run([train, metrics], feed_dict=feed_dict)
    return r_metrics[1:]


if __name__ == '__main__':
#   tf.debugging.set_log_device_placement(True)
  check_or_store_configs()
  tf.compat.v1.disable_eager_execution()
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False)) as sess:
    runner = ModelRunnerV1(sess)
    runner.train()
    runner.test()
  
  






