import tensorflow as tf


sess = tf.InteractiveSession()
a = tf.convert_to_tensor([[1, 1, 1], [2, 2, 2]])
b = a[-1:,:]
c = a[0:,:]
print(sess.run([b, c]))
sess.close()

'''
[array([[2, 2, 2]]), 
array([[1, 1, 1],
       [2, 2, 2]])]
'''


