import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape=[1]))  # 默认为一维数组，数组长度为1

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[2, 2]))  # 表示有2维数组，每个维度有两条数据

with tf.Session(graph=g1) as session:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(session.run(tf.get_variable("v")))

with tf.Session(graph=g2) as session:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(session.run(tf.get_variable("v")))
