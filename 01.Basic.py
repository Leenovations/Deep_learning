#!/usr/bin/python3

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print('\n' + '#-----------------------------------------------#' + '\n')
#-------------------------------------------------------------------------------------#
X = tf.placeholder(tf.float32, [None, 3])

x_data = [[1,2,3], [4,5,6]]
y_data = [[1,2,3], [4,5,6], [7,8,9]]

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('\n' + '#-----------------------------------------------#' + '\n')
print(x_data)
print('\n' + '#-----------------------------------------------#' + '\n')
print(sess.run(W))
print('\n' + '#-----------------------------------------------#' + '\n')
print(sess.run(b))
print('\n' + '#-----------------------------------------------#' + '\n')
print(sess.run(expr, feed_dict={X: x_data}))
sess.close()
