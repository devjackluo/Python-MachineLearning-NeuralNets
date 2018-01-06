import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

print(result)

# sess = tf.Session()
# print(sess.run(result))
# sess.close()

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)
