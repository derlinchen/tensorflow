# coding=utf-8

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')



