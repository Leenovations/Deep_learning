#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#-----------------------------------------------------------------------#
data = np.loadtxt('./data.csv',
                  delimiter=',',
                  unpack=True,
                  dtype='float32')

Feather = np.transpose(data[0:2])
Species = np.transpose(data[2:])
#-----------------------------------------------------------------------#
global_step = tf.Variable(0, trainable=False, name='global_step')
#-----------------------------------------------------------------------#
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#-----------------------------------------------------------------------#
W1 = tf.Variable(tf.random_uniform([2,10], -1, 1))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1, 1))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1, 1))
model = tf.matmul(L2, W3)
#-----------------------------------------------------------------------#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                              logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step)
#-----------------------------------------------------------------------#
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
#-----------------------------------------------------------------------#
ckpt = tf.train.get_checkpoint_state('/labmed/08.DL/model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())# # #-----------------------------------------------------------------------#
# for step in range(2):
#     sess.run(train_op, feed_dict={X:Feather, Y:Species})

#     print('Step : %d' % sess.run(global_step),
#           'Cost : %3f' % sess.run(cost, feed_dict={X:Feather, Y:Species}))
# #-----------------------------------------------------------------------#
# saver.save(sess, './model/dnn.ckpt', global_step=global_step)
# #-----------------------------------------------------------------------#
# prediction = tf.argmax(model, 1)
# target = tf.argmax(Y, 1)
# 
# print('X : ', sess.run(prediction, feed_dict={X:Feather}))
# print('Y : ', sess.run(target, feed_dict={Y:Species}))
# #-----------------------------------------------------------------------#
# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X:Feather, Y:Species}))