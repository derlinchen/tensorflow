# coding=utf-8

# 自定义损失函数

import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()

# 比较变量中的值，获取较大的值
print(tf.greater(v1, v2).eval())
print(tf.where(tf.greater(v1, v2), v1, v2).eval())

sess.close()
