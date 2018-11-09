import tensorflow as tf
import numpy as np
import random


x = tf.placeholder(tf.float32, shape=[2, 1])
y_c = tf.placeholder(tf.float32, shape=[1, 1])

W = tf.Variable(tf.zeros([2, 2]), dtype=tf.float32)
bx = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32)
h = tf.matmul(W, x) + bx
h_ = tf.nn.sigmoid(h)

V = tf.Variable(tf.zeros([1, 2]), dtype=tf.float32)
bh = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)
y = tf.matmul(V, h_) + bh
y_ = tf.nn.tanh(y)

loss = tf.square(y_ - y_c)
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


if __name__ == "__main__":
    x_data = [[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]]
    y_data = [[[0]], [[1]], [[1]], [[0]]]
    for i in range(1000):
        index = random.randint(0, 3)
        sess.run(train, feed_dict={x: x_data[index], y_c: y_data[index]})

    for i in range(4):
        print(sess.run(y_, feed_dict={x: x_data[i]}))
