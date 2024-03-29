import tensorflow as tf
import numpy as np

embed_size = 128
hidden_size = embed_size
vocab_size = 15

class YLSTMCell():
  
  def __init__(self, forget_bias=0.0, activation=tf.nn.tanh):
    self.forget_bias = forget_bias
    self.activation = activation
    ''' w shape: [256=128*2, 512=128*4] '''
    self.w = tf.Variable(tf.random.uniform([2 * embed_size, 4 * embed_size], minval=-0.1, maxval=0.1, seed=18))
    ''' b shape: [1, 512=128*4] '''
    self.b = tf.Variable(tf.random.uniform([1, 4 * embed_size], minval=-0.1, maxval=0.1, seed=17))
  
  def ACall(self, x, state):
    """
    Long short-term memory cell (LSTM)
    @param: inputs (batch,n)
    @param state: the states and hidden unit of the two cells
    """
    ''' inputs:[1,128] '''
    ''' c:[1,128], h:[1,128] '''
    c, h = state
    # linear_input = tf.concat([inputs, h], 0)
    # ''' linear_input:[2,128] '''
    linear_input = tf.concat([x, h], 1)
    ''' linear_input:[1,256] '''
    
    # ''' c:[1,128], h:[1,128] '''
    ''' w shape: [256=128*2, 512=128*4] '''
    res = tf.matmul(linear_input, self.w)
    ''' res shape: [1, 512] '''
    res = tf.add(res, self.b)
    ''' res shape: [1, 512] '''
    i, j, f, o = tf.split(value=res, num_or_size_splits=4, axis=1)
    ''' i shape: [1, 128] '''
    ''' j shape: [1, 128] '''
    ''' f shape: [1, 128] '''
    ''' o shape: [1, 128] '''
    '''
    compute cell
    '''
    new_c1 = (c * tf.nn.sigmoid(f + self.forget_bias) + 
             self.activation(j) * tf.nn.sigmoid(i))
    '''
    compute h
    '''
    new_h1 = self.activation(new_c1) * tf.nn.sigmoid(o)
    return (new_c1, new_h1)

class RNN(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.W = tf.Variable(tf.random.uniform([embed_size,embed_size], minval=-0.1, maxval=0.1, seed=1))
    self.W2 = tf.Variable(tf.random.uniform([embed_size,embed_size], minval=-0.1, maxval=0.1, seed=11))
    self.b = tf.Variable(tf.random.uniform([1,embed_size], minval=-0.1, maxval=0.1, seed=2))

  def ACall(self, x, h):
    y_temp = tf.matmul(x, self.W)
    y_temp2 = tf.matmul(h, self.W2)
    y_tmp = tf.add(y_temp, y_temp2)
    y_tmp2 = tf.add(y_tmp, self.b)
    # print("y_tmp2 shape:" + str(tf.shape(y_tmp2)))
    new_h = tf.tanh(y_tmp2)
    return new_h

class Classify(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.V = tf.Variable(tf.random.uniform([embed_size,hidden_size], minval=-0.1, maxval=0.1, seed=3))
    
  def BCall(self, y):
    logit = tf.matmul(y, self.V)
    return logit

class TrainTest(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.X = tf.Variable(tf.random.uniform([vocab_size, embed_size], minval=-0.1, maxval=0.1, seed=4))
    self.c0 = tf.Variable(tf.random.uniform([1,embed_size], minval=-0.1, maxval=0.1, seed=29), trainable=True)
    self.h0 = tf.Variable(tf.random.uniform([1,embed_size], minval=-0.1, maxval=0.1, seed=20), trainable=True)
    # self.fc1 = RNN();
    self.fc1 = YLSTMCell();
    self.cfy = Classify();
    self.optimizer = tf.optimizers.Adam()
  
  def TrainTest(self, data, to_train, topk):
    total_num = 0
    total_correct = 0
    total_loss = 0
    ''' iterate every sentence '''
    for i in range(data.shape[0]):
      with tf.GradientTape() as tape:
        loss = 0.0
        c = self.c0
        h = self.h0
        ''' iterate every token in sentence '''
        for j in range(data.shape[1]-1):
          total_num += 1
          x = data[i][j]
          label = data[i][j+1]
          x_embed = self.X[x]
          # print("x_embed shape:" + str(tf.shape(x_embed)))
          good_shape_x_embed = tf.expand_dims(x_embed, axis=0)
          # print("good_shape_x_embed shape:" + str(tf.shape(good_shape_x_embed)))
          x = good_shape_x_embed
          # new_h = self.fc1.ACall(x, h)
          new_c, new_h = self.fc1.ACall(x, (c, h))
          logit = self.cfy.BCall(new_h)
          p = tf.nn.softmax(logit)
          # m = tf.argmax(p,1)
          _, m = tf.nn.top_k(p, topk)
          m = tf.squeeze(m, axis=0)
          # print("m shape:" + str(tf.shape(m)))
          loss += tf.nn.sparse_softmax_cross_entropy_with_logits([label], logit)
          if label in m:
            total_correct += 1
          c = new_c
          h = new_h
          # print(m)
        total_loss += loss
      if (to_train == True):
        grads = tape.gradient(loss, self.trainable_variables)
        applies = []
        for (grad, var) in zip(grads, self.trainable_variables):
          if grad is not None:
            applies.append((grad, var))
        self.optimizer.apply_gradients(applies)
      # print(loss)
    info = "train" if to_train == True else "test"
    print(info + "#total_loss:" + str(total_loss) + "#accuracy:" + str(total_correct / total_num))      

train_test = TrainTest()
train_data = [ [ 6,  6,  4,  4, 10, 10,  5,  4,  5,  3],
               [10,  1,  8,  1,  1,  9, 10,  5,  7,  6],
               [ 4,  8,  2,  2,  7,  1, 10, 10,  5,  7],
               [ 4,  6,  2,  6, 10,  8,  2,  4, 10,  7],
               [10,  6,  6,  2,  9,  3,  9, 10,  5,  2],
               [ 4,  1,  4,  6,  8,  8,  5,  6,  2,  1],
               [ 5,  5,  4,  7,  2,  7,  9,  2,  5,  7],
               [ 9,  4,  3,  2,  4,  1,  4,  7,  3,  6],
               [ 5,  7, 10,  7,  6,  5,  5,  3,  2,  9],
               [ 9,  9,  8,  3,  2,  3,  3,  7,  7,  5] ]
train_data = np.array(train_data)

test_data = [[10,  1, 10,  3,  4],
             [ 7,  5,  6,  9,  5],
             [ 6,  1,  9,  3,  6],
             [ 5,  4,  6,  9,  1],
             [ 5,  6,  8, 10,  4]]
test_data = np.array(test_data)


for i in range(50):
  print("=== " + str(i) + "th turn ===")
  ''' training '''
  train_test.TrainTest(train_data, True, 5)
  ''' testing '''
  train_test.TrainTest(test_data, False, 5)





