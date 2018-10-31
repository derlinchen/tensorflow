# coding=utf-8

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# placeholder表示占用了一个数据位置，主要用于在运行时指定要执行的数据
# x = tf.placeholder(tf.float32, shape=(1,2),name='input')
# 也可以是多行矩阵
x = tf.placeholder(tf.float32, shape=(3, 2), name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 通过feed_dict向placeholder指定要传入的数据，并且做相关的计算
# print(sess.run(y, feed_dict={x:[[0.7,0.9]]}))
# 多行矩阵的情况
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.6, 0.8], [0.8, 0.7]]}))
