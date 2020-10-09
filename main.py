from __future__ import print_function
from readFile import getTrainData, getTestData

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import data
import readFile


# print(tf.__version__)

# Import MNIST data(下载数据)
# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
data = data.Data()

# 参数：学习率，训练次数，
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 40000])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.int32, [None, 1])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([40000, 1]))
b = tf.Variable(tf.zeros([1]))

# softmax模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy（损失函数用cross entropy）
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# Gradient Descent（梯度下降优化）
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(data.getTrainSize() / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = data.nextTrainBatch()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: readFile.getTrainData()[0].values(), y: readFile.getTrainData()[1].values()}))
