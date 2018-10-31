# coding=utf-8

import tensorflow as tf

# 数据类型不一致
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1), name='w1')
# w2 = tf.Variable(tf.random_normal([2,3], dtype=tf.float64, stddev=1), name='w2')
# w1.assign(w2)

# w3 = tf.Variable(tf.random_normal([2,3],stddev=1), name='w3')
# w4 = tf.Variable(tf.random_normal([2,2],stddev=1),name='w4')
# 维度不匹配
# tf.assign(w3, w4)

# 不验证维度匹配
# tf.assign(w3, w4, validate_shape=False)

