'''

    Do machine learning by tensorflow

'''

import tensorflow as tf
from numpy.random import RandomState
import numpy as np

#   定义训练数据 batch 的大小
batch_size = 2

#   定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 1], mean=0,stddev = 1,seed=1))
d1 = tf.Variable(tf.random_normal([1, 1], mean=0,stddev = 1,seed=1))


x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#   定义神经网络前向传播的过程
y = tf.matmul(x, w1)+d1


#   定义损失函数和反向传播的算法
y = tf.sigmoid(y)


cross_entropy = -tf.reduce_mean(y_*tf.log(y)+(1-y_)*tf.log(1-y))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

X =np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]).astype(np.float32)

#   标签
Y = np.array([[0.0],[1.0],[1.0],[1.0],[1.0],[1.0],[0.0],[1.0],[1.0]]).astype(np.float32)#or

X_test=np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ]).astype(np.float32)


init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        for i in range(len(X)-2):
            sess.run(train_step,feed_dict={x:X[i:i+2],y_:Y[i:i+2]})
    print("Test:",sess.run(y,feed_dict={x:X_test}))




