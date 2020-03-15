from utils.config_util import check_or_store_configs
from models.model_runner import ModelRunner


if __name__ == '__main__':
#   tf.debugging.set_log_device_placement(True)
#   tf.compat.v1.disable_eager_execution()
  check_or_store_configs()
  runner = ModelRunner()
  runner.train()
  runner.test()
  
  

