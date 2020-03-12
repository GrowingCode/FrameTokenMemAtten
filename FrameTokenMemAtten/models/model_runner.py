


if __name__ == '__main__':
  self.metrics = f_res[2:]
  if training:
    self.train = self.clip_and_apply_gradients(self.metrics[self.metrics_index["all_loss"]])
  else:
    self.train = tf.constant(0, int_type)
