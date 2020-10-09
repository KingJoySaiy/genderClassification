# linear regression(1 dimension) using tensorflow

import numpy as np
import tensorflow as tf

# training data
xTrain = [1, 2, 3, 6, 8]
yTrain = [4.8, 8.5, 10.4, 21, 25.3]

# parameter
W = tf.Variable([np.random.rand()], dtype=tf.float32)
b = tf.Variable([np.random.rand()], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# predict & loss function
predict = W * x + b
loss = tf.reduce_sum(tf.square(predict - y))

# new process
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # need to initialize

# print(sess.run(predict, {x: dataX}))
# print(sess.run(loss, {x: dataX, y: dataY}))

# gradient optimizer (learning rate == 0.001)
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

for i in range(10000):
    sess.run(train, {x: xTrain, y: yTrain})
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b), sess.run(loss, {x: xTrain, y: yTrain})))
