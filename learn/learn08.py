# coding=utf-8

import tensorflow as tf

# numpy 为一个科学计算的工具包，通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义了训练数batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维护上使用None可以方便使用不大的batch。在训练是需要把数据分成较小的batch
# 但在测试是，可以一次性使用全部的数据。当数据集比较小的时候，这样比较方便测试，但数据集比较大是，将一个大量数据放入batch可能会导致内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络钱箱传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本标签，x1 + x2 < 1的样例被认为正样本，其他为负样本，这里使用0表示负样本，1表示正样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow程序
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 训练之前神经网络参数的值
    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X,y_:Y})
            print("after %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
