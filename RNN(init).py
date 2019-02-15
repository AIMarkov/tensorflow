import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
class RNN:
    #激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __init__(self,NO_OF_INPUTS,INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE ):
        # 10 parameters per input, morphed to 100 from input layer, gives one value as output
        self.NO_OF_INPUTS = NO_OF_INPUTS     #输入数据的个数
        self.INPUT_SIZE = INPUT_SIZE         #输入数据的维度
        self.HIDDEN_SIZE = HIDDEN_SIZE       #隐藏层神经元个数
        self.OUTPUT_SIZE = OUTPUT_SIZE       #输出层个数
        # Input and Output values X and Y assigned at random
        #self.X = np.random.randint(0, 2, (self.NO_OF_INPUTS, self.INPUT_SIZE))
        #self.Y = np.random.randint(0, 2, (self.NO_OF_INPUTS, self.OUTPUT_SIZE))

        # RNN Forward Prop Values
        self.iv = np.random.uniform(-0.01, 0.01, size=(self.NO_OF_INPUTS, self.HIDDEN_SIZE))        #输入x和权重U相乘后的结果
        self.RNN_hv = np.random.uniform(-0.01, 0.01, size=(self.NO_OF_INPUTS, self.HIDDEN_SIZE))    #循环层状态s的结果
        self.ov = np.random.uniform(-0.01, 0.01, size=(self.NO_OF_INPUTS, self.OUTPUT_SIZE))        #输出的结果  10维度
        # RNN Weights

        self.W_ih = np.random.uniform(-0.01, 0.01, size=(self.INPUT_SIZE, self.HIDDEN_SIZE))        #权重U
        self.RNN_W_hh = np.random.uniform(-0.01, 0.01, size=(self.HIDDEN_SIZE, self.HIDDEN_SIZE))   #权重W
        self.W_ho = np.random.uniform(-0.01, 0.01, size=(self.HIDDEN_SIZE, self.OUTPUT_SIZE))       #权重v
        # RNN Back Prop Values

        self.RNN_dov = np.random.uniform(-0.01, 0.01, size=(self.NO_OF_INPUTS, self.OUTPUT_SIZE))   #输出和标签之间的误差项
        self.RNN_dhv = np.random.uniform(-0.01, 0.01, size=(self.NO_OF_INPUTS, self.HIDDEN_SIZE))   #循环层W误差项
        self.RNN_dhvprev = self.RNN_dhv                                                             #记录上一时刻的误差项
        self.RNN_div = np.random.uniform(-0.01, 0.01, size=(self.NO_OF_INPUTS, self.HIDDEN_SIZE))   #循环层U误差项
        self.RNN_hvprev = self.RNN_hv                                                               #记录上一时刻的循环层的状态的值

        self.state_list=[]
        self.Times=0
        self.delta_list=[]

#前向计算
    def RNN_forward(self,X):
        self.Times+=0
        self.iv = np.dot(X, self.W_ih)
        #self.RNN_hv = np.tanh(self.iv + np.dot(self.RNN_hvprev, self.RNN_W_hh))
        self.RNN_hv = self.sigmoid(self.iv + np.dot(self.RNN_hvprev, self.RNN_W_hh))

        self.ov = np.dot(self.RNN_hv, self.W_ho)
        self.RNN_hvprev = self.RNN_hv
        self.state_list.append(self.RNN_hv)
        return self.ov

#反向传播
    def RNN_backprop(self,X,Y):
        self.RNN_dov = Y - self.ov

        self.RNN_dhv = (self.RNN_dov.dot(self.W_ho.T) + self.RNN_dhvprev.dot(self.RNN_W_hh.T)) * (1 - self.RNN_hv * self.RNN_hv)

        self.RNN_div = (self.RNN_dov.dot(self.W_ho.T)) * (1 - self.RNN_hv * self.RNN_hv)

        self.W_ho += self.RNN_hv.T.dot(self.RNN_dov) * 0.01
        self.RNN_W_hh += self.RNN_hvprev.T.dot(self.RNN_dhv) * 0.01
        self.W_ih += X.T.dot(self.RNN_div) * 0.01
        self.delta_list.append(self.RNN_dhv)
        self.RNN_dhvprev = self.RNN_dhv                                                            #记录上一时刻循环层的误差项

if __name__ == '__main__':
    training_iters=2000
    model = RNN(1,784,100,10)
    mnist = input_data.read_data_sets("Mnist_Data", one_hot=True)
    for step in range(training_iters):
        batch_X = []
        batch_Y = []
        count = 0
        for i in range(10):
            batch_xs, batch_ys = mnist.train.next_batch(1)
            batch_X.append(batch_xs)
            batch_Y.append(batch_ys)

        for i in range(len(batch_X)):
            result=model.RNN_forward(batch_X[i])
            result_index = np.argmax(result)
            label_index = np.argmax(batch_Y[i])
            if result_index == label_index:
                count += 1
            model.RNN_backprop(batch_X[i], batch_Y[i])
        ac = float(count / len(batch_Y))
        if step%10==0:
            print("Iter " + str(step) + ", Testing Accuracy=" + str(ac))
    count1=0
    for i in range(10000):

        batch_xs, batch_ys = mnist.train.next_batch(1)
        result = model.RNN_forward(batch_xs)
        result_index = np.argmax(result)
        label_index = np.argmax(batch_ys)
        if result_index == label_index:
            count1 += 1
    print("testing Accuracy="+str(count1/10000))


