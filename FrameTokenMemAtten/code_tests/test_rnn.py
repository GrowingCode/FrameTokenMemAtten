import tensorflow as tf


embed_size = 2
hidden_size = 3
vocab_size = 10

class FullConn(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.W = tf.Variable(tf.random.uniform([embed_size,embed_size], minval=-0.1, maxval=0.1, seed=1))
    self.b = tf.Variable(tf.random.uniform([1,embed_size], minval=-0.1, maxval=0.1, seed=2))
  
  '''
  y = tanh(x * W + b)
  '''
  def ACall(self, x):
    '''
    x: [1, 2]
    '''
    y_temp = tf.matmul(x, self.W)
    y_tmp = tf.add(y_temp, self.b)
    y = tf.tanh(y_tmp)
    
    return y


class Classify(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.V = tf.Variable(tf.random.uniform([embed_size,hidden_size], minval=-0.1, maxval=0.1, seed=3))
    
  '''
  logit = tanh(y * V)
  '''
  def BCall(self, y):
    '''
    x: [1, 2] => 0
    label:[0, 1, 2]
    0: 猫
    1: 狗
    2: 兔
    '''
    logit = tf.matmul(y, self.V)
    return logit

class TrainTest(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.X = tf.Variable(tf.random.uniform([vocab_size, embed_size], minval=-0.1, maxval=0.1, seed=4))
    self.fc1 = FullConn();
    self.fc2 = FullConn();
    self.fc3 = FullConn();
    self.cfy = Classify();
    self.optimizer = tf.optimizers.Adam()
    
    print(self.trainable_variables)
  
  def Train(self, x, label):
    with tf.GradientTape() as tape:
      ''' x_embed shape: [2] '''
      x_embed = self.X[x]
      print(tf.shape(x_embed))
      good_shape_x_embed = tf.expand_dims(x_embed, axis=0)
      print(tf.shape(good_shape_x_embed))
      x = good_shape_x_embed
      ''' begin '''
      x1 = self.fc1.ACall(x)
      x2 = self.fc2.ACall(x1)
      x3 = self.fc3.ACall(x2)
      logit = self.cfy.BCall(x3)
      p = tf.nn.softmax(logit)
      ''' end '''
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits([label], logit)
      
      grads = tape.gradient(loss, self.trainable_variables)
      applies = []
      for (grad, var) in zip(grads, self.trainable_variables):
        if grad is not None:
          applies.append((grad, var))
      self.optimizer.apply_gradients(applies)
    
    print(p)
    # print(loss)

train_test = TrainTest()
''' [1,2] '''
x = 1
label = 0

for _ in range(1):
  train_test.Train(x, label)



