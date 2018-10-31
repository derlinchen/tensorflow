# coding=utf-8

import tensorflow as tf

# 两个张量的数据类型不一致时，在下例中会出现计算错误的异常， 故需想张量指定数据类型：dtype="float32" 或 dtype=tf.float32
a = tf.constant([1, 2], name="a", dtype="float32")
a = tf.constant([1, 2], name="a", dtype=tf.float32)
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result)
