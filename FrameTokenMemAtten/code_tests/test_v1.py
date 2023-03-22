import tensorflow as tf


# tf.compat.v1.disable_eager_execution()

x = tf.constant([5, 4, 6])
y = tf.constant([5, 2, 5])
# x = tf.compat.v1.placeholder(tf.float32, [3])
# y = tf.compat.v1.placeholder(tf.float32, [3])
z = tf.math.greater(x, y)

if (z[0]):
  v = x+y;
else :
  v = x-y;

# sess = tf.compat.v1.Session()
# z = sess.run(z, feed_dict={x:[5,4,6], y:[5, 2, 5]})
# v = sess.run(v, feed_dict={x:[5,4,6], y:[5, 2, 5]})

print(v)

if (z[0]):
  print("z[0] is True")
else:
  print("z[0] is False")

if (z[1]):
  print("z[1] is True")
else:
  print("z[1] is False")



