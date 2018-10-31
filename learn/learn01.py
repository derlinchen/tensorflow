# coding=utf-8

import tensorflow as tf

a = tf.constant([1.0,2.0], name="a")
b = tf.constant([2.0,3.0], name="b")
result = a + b

# name-名字，表示这个张量是如何计算的
# shape-维度，表示一个一维数组，长度为2,shape第一个参数表示数据长度，第二个参数表示维度
# dtype-表示数据类型
print(result)

print(a.graph is tf.get_default_graph())