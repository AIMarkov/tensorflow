# -*- coding: UTF-8 -*-
from __future__ import  print_function
import numpy as np


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

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
        self.W = np.random.uniform(-0.1, 0.1,
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
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))


# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
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
               self.train_one_sample(labels[d],
                    data_set[d], rate)


    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):

       delta = self.layers[-1].activator.backward(
            self.layers[-1].output
       ) * (label - self.layers[-1].output)
       for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
       return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg))
        , args
    ))
# 数据加载器基类
class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count
    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    content[start + i * 28 + j])
        return picture
    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set

# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels
    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec

def get_training_data_set():
        '''
        获得训练数据集
        '''
        image_loader = ImageLoader('E:\lar\deep\MNIST_data\\train-images.idx3-ubyte', 10)
        return image_loader.load()

def get_training_labels():
        label_loader = LabelLoader('E:\lar\deep\MNIST_data\\train-labels.idx1-ubyte', 10)
        return label_loader.load()

def get_test_data_set():
        image_loader = ImageLoader('E:\lar\deep\MNIST_data\\t10k-images.idx3-ubyte', 10)
        return image_loader.load()

def get_test_labels():
    label_loader = LabelLoader('E:\lar\deep\MNIST_data\\t10k-labels.idx1-ubyte', 10)
    return  label_loader.load()

def train_data_set():
    data_set = []
    labels = []
    for i in range(10):
         data_set.append(get_training_data_set()[i])
         labels.append(get_training_labels()[i])
    return data_set,labels

def test_data_set():
    data_set = []
    labels = []
    for i in range(10):
         data_set.append(get_test_data_set()[i])
         labels.append(get_test_labels()[i])
    return data_set,labels


def train_eval():
        data_set,labels = transpose(train_data_set())
        #for i in range(6000):
         #   print(labels[i],data_set[i])
        net = Network([784, 200, 10])
        rate = 0.5
        mini_batch = 20
        epoch = 15
        for i in range(epoch):
            net.train(labels, data_set, rate, mini_batch)
            loss_value=net.loss(labels[-1], net.predict(data_set[-1]))
            if loss_value<0.0000001:
                break
            print ('after epoch %d loss: %f' % (
                (i + 1),loss_value
            ))

        correct_num=0
        data_set,labels = transpose(test_data_set())
        for j in range(10):
            if np.argmax(net.predict(data_set[j])) == np.argmax(labels[j]):
                correct_num += 1.0
        corrct_ratio=correct_num/10.0
        print(corrct_ratio)

train_eval()