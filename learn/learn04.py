# coding=utf-8

# 张量的中间计算结果引用
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b

# 直接计算向量的和
res = tf.constant([2.0, 3.0], name="a") + tf.constant([2.0, 3.0], name="b")

print(result)
print(res)


# Session的创建方式
try:
    sess = tf.Session()
    print(sess.run(result))
except Exception as e:
    print(e)
finally:
    sess.close()

# 使用with的方式，session会自动关闭
try:
    with tf.Session() as session:
        print(session.run(res))
except Exception as e:
    print(e)

# 通过session.as_default()方法来定义默认会话，定义默认会话后可通过eval来计算张量的值
session = tf.Session()
with session.as_default():
    print(res.eval())

# 形式也如下
sess = tf.Session()
print(sess.run(result))
print(result.eval(session=sess))
sess.close()

# 也可使用tf.InteractiveSession来构建默认session
sess = tf.InteractiveSession()
print(res.eval())
sess.close()

# allow_soft_placement表示定义GPU分配策略，是否可以在GPU上运行
# log_device_placement 表示在日志中记录每个节点的信息
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
ss = tf.InteractiveSession(config=config)
ss2 = tf.Session(config=config)
print(ss.run(result))
print(ss2.run(res))

