#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))  #正向传播直接向量
    def backward(self, output):
        return output * (1 - output)  #反向传播没有向量

# 获取卷积区域
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
            start_i: start_i + filter_height,
            start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
            start_i: start_i + filter_height,
            start_j: start_j + filter_width]


# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    width = array.shape[1]
    n = np.argmax(array)
    i = n // width
    j = n % width
    return i, j


# 为数组增加Zero padding
def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    '''
    zp = int(zp)
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:, zp: zp + input_height, zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array


# 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        i[...] = op(i)


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1, 1,
                                         (depth, height, width))
        self.bias = 0
        self.weights_grad = np.ones(
            self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding,
                stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding,
                stride)
        self.output_array = np.zeros((self.filter_number,
                                      self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,
                                          self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array,
                        self.activator.forward)
        return self.output_array

    def backward(self, input_array, sensitivity_array,
                 activator):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        self.bp_sensitivity_map(sensitivity_array,
                                activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(list(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights())))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array,
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,
                         self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size,
                              filter_size, zero_padding, stride):
        return int((input_size - filter_size +
                2 * zero_padding) / stride + 1)


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height, self.output_width))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())
        return self.output_array

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                                     i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-1, 1,
            (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))


# 神经网络类
class Network(object):
    def __init__(self, layers, activator):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    activator
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(list(data_set))):
               self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
       # print(self.layers[-1].output)
       delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
       for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()


# 手工读取数据
def init():
    file1 = 'E:/PycharmProjects/tensorflow_CNN/MNIST_data/train-images.idx3-ubyte'
    file2 = 'E:/PycharmProjects/tensorflow_CNN/MNIST_data/train-labels.idx1-ubyte'
    imgs, data_head = Image.loadImageSet(file1)
    labels, labels_head = Image.loadLabelSet(file2)
    return imgs, labels

'''
# 偏置量
def get_bias(shape):
    num = 1
    for i in shape:
        num *= i
    initial = np.linspace(0.1, 0.1, num).reshape(shape)
    return initial
# 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
def get_weight(shape):
    initial = np.random.normal(0, 0.1, shape)
    return initial
'''

# 计算卷积
def conv(input_array,
         kernel_array,
         output_array,
         stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (get_patch(input_array, i, j, kernel_width, kernel_height,
                                            stride) * kernel_array).sum() + bias


def softmax(x):
    prediction = np.zeros([np.size(x)], dtype='float')
    exp_sum = 0.
    for i in range(np.size(x)):
        exp_sum += np.exp(x[i])
    for i in range(np.size(x)):
        prediction[i] = float(np.exp(x[i]) / exp_sum)
    return prediction


def loss(label, output):
    return 0.5 * ((label - output) * (label - output)).sum()


if __name__ == "__main__":

    np.set_printoptions(threshold=np.nan)
    # 激活函数
    activator = SigmoidActivator()
    training_iters = 2000
    rate = 0.6
    # 两个卷积层
    model_conv1 = ConvLayer(28, 28, 1, 5, 5, 32, 2, 1, activator, rate)
    model_conv2 = ConvLayer(14, 14, 32, 5, 5, 64, 2, 1, activator, rate)
    # 两个pooling层
    max_pool_model1 = MaxPoolingLayer(28, 28, 32, 2, 2, 2)
    max_pool_model2 = MaxPoolingLayer(14, 14, 64, 2, 2, 2)
    # 全连接层
    net = Network([7*7*64, 1024, 10], activator)

    mnist = input_data.read_data_sets("Mnist_Data", one_hot=True)

    batch_X = []
    batch_Y = []

    for i in range(50):
        batch_xs, batch_ys = mnist.train.next_batch(1)
        batch_X.append(batch_xs)
        batch_Y.append(batch_ys)

    for step in range(training_iters):
        count = 0
        for i in range(len(batch_X)):
            img = np.array(batch_X[i]).reshape([1, 28, 28])        # 28*28
            # conv 1
            result_conv1 = model_conv1.forward(img)         # 32*28*28
            # max pooling 1
            result_pool1 = max_pool_model1.forward(result_conv1)    # 32*14*14
            # conv 2
            result_conv2 = model_conv2.forward(result_pool1)        # 64*14*14
            # max pooling 2
            result_pool2 = max_pool_model2.forward(result_conv2)     # 64*7*7

            label_index = np.argmax(batch_Y[i])
            result_pool3 = result_pool2.reshape([7*7*64, 1])
            # 全连接
            net.train_one_sample(label_index, result_pool3, rate)
            print(net.layers[1].output)
            # softmax回归
            prediction = softmax(net.layers[1].output)
            print(prediction)
            # 预测
            # print(prediction)
            result_index = np.argmax(prediction)
            if result_index == label_index:
                count += 1
            print(i)
            print(label_index, result_index)
            print('------------------------------------------------------------------------')
            max_pool_model2.backward(result_conv2, net.layers[0].delta.reshape([64, 7, 7]))
            model_conv2.backward(result_pool1, max_pool_model2.delta_array, activator)
            model_conv2.update()
            max_pool_model1.backward(result_conv1, model_conv2.delta_array)
            model_conv1.backward(img, max_pool_model1.delta_array, activator)
            model_conv1.update()

        ac = float(count / len(batch_Y))
        print('----------step:', step, '-------------')

        print("Iter " + str(step) + ", Testing Accuracy=" + str(ac))


