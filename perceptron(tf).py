#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

#可以在运行图的时候插入一些计算图，适用于交互式
sess = tf.InteractiveSession()
# 第一个None是表示批量的样本输入
x = tf.placeholder("float", shape=[None, 784])

# 第一个None表示批量的样本标签
y_ = tf.placeholder("float", shape=[None, 10])

# 只是一个层，W有784*10个变量
W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))
# 初始化变量
sess.run(tf.initialize_all_variables())

#更新公式
a = tf.matmul(x, W) + b
y = tf.nn.softmax(a)

# y_是标签给的，y是我们预测的，y_ * tf.log(y)是向量对应的元素相乘，
# 结果仍是一个和原来维数一致的向量
# reduce_sum是把原向量值相加成一个数
corss_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(corss_entropy)

# correct_pre返回布尔值的list,argmax(y, 1)1是找的每一行的最大值下标，0是列,返回的都是List
correct_pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, "float"))

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    ac = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
    if i % 10 == 0:
        print("Iter " + str(i) + ", Training Accuracy = " + str(ac))

ac = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Testing Accuracy: " + str(ac))
sess.close()

