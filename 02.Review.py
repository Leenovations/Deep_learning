#!/usr/bin/python3

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('\n' + '#-----------------------------------------------#' + '\n')
#---------------------------------------------------------------------------------------#
x = [1,2,3]
y = [3,6,9]
#---------------------------------------------------------------------------------------#
W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))
#---------------------------------------------------------------------------------------#
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
#---------------------------------------------------------------------------------------#
hypothesis = W * X + b
#---------------------------------------------------------------------------------------#
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)
#---------------------------------------------------------------------------------------#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x, Y:y})
        print(step, cost_val, sess.run(W), sess.run(b))

    print(sess.run(hypothesis, feed_dict={X:10}))