# coding=utf-8

import tensorflow as tf

# 声明两个变量，通过seed设定随机种子，保证每次运行得到的结果一样
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 将输入的特征向量定义为一个常量
x = tf.constant([[0.7, 0.9]])

# 向前传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# w1与w2定义的时候没有初始化，需通过initializer进行初始化后才可以进行计算
# initializer进行单个变量的初始化

# sess.run(w1.initializer)
# sess.run(w2.initializer)

# 对所有的变量进行初始化,使用global_variables_initializer方法
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y))
sess.close()
