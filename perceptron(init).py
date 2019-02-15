#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from functools import reduce
from numpy import *
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# 感知器类的实现
class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化
        self.weights = np.zeros([input_num,10])
        # 偏置项初始化
        self.bias = np.zeros([10])

    def predict(self, input_vec):           #对应公式(2)-(6)
        '''
        输入向量，输出感知器的计算结果
        '''
        return self.activator(np.dot(input_vec,self.weights) + self.bias)

    def train(self, input_vecs, labels, rate):
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = list(zip(input_vecs, labels))
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        # 按照感知器规则更新权重
        delta = label - output
        self.weights = list(map(            #对应公式(14)
            lambda x, w: w + rate * delta * x,
            input_vec, self.weights))
        # 更新bias
        self.bias += rate * delta           #对应公式(15)

# 利用sigmoid函数实现感知器
def f(x):
    return 1/(1 + np.exp(-x))

if __name__ == '__main__':
    # 训练感知器
    perceptron = Perceptron(784,f)        #28*28
    training_iters = 1000
    mnist = input_data.read_data_sets("Mnist_Data", one_hot=True)       #得到mnist数据
    for step in range(training_iters):
        batch_X = []
        batch_Y = []
        count = 0   #用来记录标签正确的数据个数
        # 取接下来的10个训练数据和标签
        for i in range(10):
            batch_xs, batch_ys = mnist.train.next_batch(1)
            batch_X.append(batch_xs)
            batch_Y.append(batch_ys)

        for i in range(len(batch_X)):   #对于其中的每一个数据，都计算得到预测值
            result = perceptron.predict(batch_X[i])
            result_index = np.argmax(result)        #得到result中值最大的下标
            label_index = np.argmax(batch_Y[i])     #得到相应标签中值最大的下标
            if result_index == label_index:         #如果相等，说明预测正确，count+1
                count += 1
            perceptron.train(batch_X[i], batch_Y[i], 0.01)        #更新权重
        ac = float(count / len(batch_Y))
        if step % 10 == 0:
            print("Iter " + str(step) + ", Training Accuracy = " + str(ac))

    #测试
    count = 0
    for i in range(10000):
        test_input, test_label = list(mnist.test.next_batch(1))
        pred = perceptron.predict(test_input)
        if np.argmax(pred) == np.argmax(test_label):
            count += 1

    ac = float(count/10000)
    print ("Test accuracy = " + str(ac))

