#!/usr/bin/python3

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#---------------------------------------------------------------------------------------#
x = [1,2,3]
y = [1,4,9]
#---------------------------------------------------------------------------------------#
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#---------------------------------------------------------------------------------------#
W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))
#---------------------------------------------------------------------------------------#
hypothesis = W * tf.square(X) + b
#---------------------------------------------------------------------------------------#
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
#---------------------------------------------------------------------------------------#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        sess.run(train_op, feed_dict={X:x, Y:y})

    print(sess.run(hypothesis, feed_dict={X:10}))
#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#