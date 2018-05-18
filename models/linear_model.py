import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()

# Create linear model K*X + B = Y
W = tf.Variable([.0], np.float32)  # Init value + Type value
b = tf.Variable([.0], np.float32)
x = tf.placeholder(np.float32)  # Input param type
y = tf.placeholder(np.float32)  # True result
# Result Tensor
Y = W * x + b

# Init variable with init values
sess.run(tf.global_variables_initializer())  # Init variable

# Try run session with values for x
print(sess.run(Y, {x: [1, 2, 3, 4]}))

# Calculate error
error = tf.square(Y - y)

loss = tf.reduce_sum(error)  # sum for all elements in Tensor

# Simple gradient optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
# Step down by gradient
train = optimizer.minimize(loss)

# loss massive for test display
loss_massive = []

for i in range(10000):
    sess.run(train, {x: [1,2,3,4], y: [0, -1, -2, -3]})
    if i % 100 == 0:
        loss_a = sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
        loss_massive.append(loss_a)
        print('Iteration: ' + str(i) + ' error: ' + str(loss_a))

# dump graph
# writer = tf.summary.FileWriter('./my_graph', sess.graph)
# writer.close()

plt.plot(loss_massive, 'r')
plt.show()
print(sess.run([W, b]))