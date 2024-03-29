#!/usr/bin/python3

import tensorflow as tf
import numpy as np
#-----------------------------------------------------------#
Feather = np.array([[1,0],
                    [0,1],
                    [0,0],
                    [1,1],
                    [0,0],
                    [1,1]])

Species = np.array([[0,1,0],
                    [0,0,1],
                    [1,0,0],
                    [0,0,1],
                    [1,0,0],
                    [0,0,1]])
#-----------------------------------------------------------#
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#-----------------------------------------------------------#
W1 = tf.Variable(tf.random_uniform([2,10], -1, 1))
b1 = tf.Variable(tf.zeros([10]))

W2 = tf.Variable(tf.random_uniform([10,3], -1, 1))
b2 = tf.Variable(tf.zeros([3]))

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)
#-----------------------------------------------------------#
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                              logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
#-----------------------------------------------------------#
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#-----------------------------------------------------------#
for step in range(100):
    sess.run(train_op, feed_dict={X:Feather, Y:Species})
#-----------------------------------------------------------#
prediction = tf.argmax(model, 1)
target = tf.argmax(Species, 1)

print(sess.run(prediction, feed_dict={X:Feather}))
print(sess.run(target, feed_dict={X:Species}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print(sess.run(accuracy*100, feed_dict={X:Feather, Y:Species}))
#-----------------------------------------------------------#
# Feather = np.array([[0,0],
#                     [1,0],
#                     [1,1],
#                     [0,0],
#                     [0,0],
#                     [0,1]])

# Species = np.array([[1,0,0],
#                     [0,1,0],
#                     [0,0,1],
#                     [1,0,0],
#                     [1,0,0],
#                     [0,0,1]])
# #-----------------------------------------------------------#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# #-----------------------------------------------------------#
# W1 = tf.Variable(tf.random_uniform([2,10], -1, 1))
# b1 = tf.Variable(tf.zeros([10]))

# W2 = tf.Variable(tf.random_uniform([10,10], -1, 1))
# b2 = tf.Variable(tf.zeros([10]))

# W3 = tf.Variable(tf.random_uniform([10,3], -1, 1))
# b3 = tf.Variable(tf.zeros([3]))
# #-----------------------------------------------------------#
# L1 = tf.add(tf.matmul(X, W1), b1)
# L1 = tf.nn.relu(L1)

# L2 = tf.add(tf.matmul(L1, W2), b2)

# model = tf.add(tf.matmul(L2, W3), b3)
# #-----------------------------------------------------------#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
# train_op = optimizer.minimize(cost)
# #-----------------------------------------------------------#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# for step in range(3):
#     sess.run(train_op, feed_dict={X:Feather, Y:Species})

#     if (step + 1) % 10 == 0:
#         print(step+1, sess.run(cost, feed_dict={X:Feather, Y:Species}))

# prediction = tf.argmax(model, 1)
# target = tf.argmax(Y, 1)

# print('X : ', sess.run(prediction, feed_dict={X:Feather}))
# print('Y : ', sess.run(target, feed_dict={Y:Species}))
# #-----------------------------------------------------------#
# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X:Feather, Y:Species}))
# #-----------------------------------------------------------#