import tensorflow as tf
import numpy as np


'''
Simple placeholder and constant
'''
# create session
sess = tf.InteractiveSession()

# variable
a = tf.placeholder(np.int32, name='a')
b = tf.placeholder(np.int32, name='b')
# const
d = tf.constant(10, np.int32)

# simple operation
c = d * a + b

# init global variable
tf.global_variables_initializer()

print(sess.run(c, feed_dict={a: np.array([10, 1, 2, 3, 4]), b: np.array([10, 12, 33, 44, 55])}))

# different types
a_t = tf.placeholder(np.float32)
b_t = tf.placeholder(np.int32)

# cast to type
c = a_t * tf.cast(b_t, np.float32)

print(sess.run(tf.cast(c, np.int32), feed_dict={a_t: [19], b_t: [10]}))

massive = sess.run(c, feed_dict={a_t:[1, 2, 3, -1], b_t: [10, 20, 30, 40]})
print(massive)
max = sess.run(tf.argmax(massive, axis=-1))

print(max)
print(massive[max])

# matrix mathematics
a_m = tf.placeholder(np.float32, name='a_m')
b_m = tf.placeholder(np.float32, name='b_m')
c_m = b_m * (a_m + a_m * b_m)

matrix_1 = np.array(
    [
        [1, 3, 4, 5, 6],
        [2, 4, 6, 7, 8],
        [22, 33, 33, 32, 3],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2]
    ]
)

matrix_2 = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)


sum = sess.run(c_m, feed_dict={a_m: matrix_1, b_m: matrix_2})
print(sum)
# ANN ?

