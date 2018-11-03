# coding=utf-8

import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
sess = tf.InteractiveSession()
# 对张量进行值的限定,第一个参数为传入的张量,第二个值为新张量的最小值,第三个值为新张量的最大值
print(tf.clip_by_value(v, 2.5, 4.5).eval())

v1 = tf.constant([1.0, 2.0, 3.0])
# 进行对数处理，即logN
print(tf.log(v1).eval())

# 乘法运算
v2 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v3 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
# * 执行的为1.0*5.0 2.0*6.0 3.0*7.0 4.0*8.0
print((v2 * v3).eval())
# matmul执行的为矩阵乘法运算
print(tf.matmul(v2, v3).eval())

# v4表示2个样例数量3个结果，v4计算的是这个一个batch(2个样例)的平均交叉熵，即标样
v4 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.reduce_mean(v4).eval())

# 交叉熵与softmax回归使用,y代表原始圣经网络的输出结果，y_表示给定的标准答案
# 回归问题解决的是对具体数值的预测
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)

# 回归问题的均方误差损失函数的实现,y_为标准答案, y为神经网络的输出答案
# mse = tf.reduce_mean(tf.square(y_, y))

sess.close()
