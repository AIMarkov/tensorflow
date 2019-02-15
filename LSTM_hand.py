import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("Mnist_Data",one_hot=True)
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))  #正向传播直接向量
    def backward(self, output):
        return output * (1 - output)  #反向传播没有向量
class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0
    def backward(self, output):
        return 1 - output * output


class LstmLayer(object):
    def __init__(self, input_width, state_width,
                 learning_rate):
        self.input_width = input_width  #也就是x的(input_width,1)（3,1）
        self.state_width = state_width  #也就是h的(state_width,1)（2,1）
        self.learning_rate = learning_rate
        # 门的激活函数
        self.gate_activator = SigmoidActivator()
        # 输出的激活函数
        self.output_activator = TanhActivator()
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()  #注意list里面是np数组
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()
        # 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
        self.Wfh, self.Wfx, self.bf = (
            self.init_weight_mat())
        # 输入门权重矩阵Wih, Wix, 偏置项bi
        self.Wih, self.Wix, self.bi = (
            self.init_weight_mat())
        # 输出门权重矩阵Woh, Wox, 偏置项bo
        self.Woh, self.Wox, self.bo = (
            self.init_weight_mat())
        # 单元状态c'权重矩阵Wch, Wcx, 偏置项bc
        self.Wch, self.Wcx, self.bc = (
            self.init_weight_mat())

    def init_state_vec(self):
        '''
        初始化保存状态的向量
        '''
        state_vec_list = []
        state_vec_list.append(np.zeros(
            (self.state_width, 1)))
        return state_vec_list

    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh = np.random.uniform(-1e-4, 1e-4,
                               (self.state_width, self.state_width))  #随机初始化权重
        Wx = np.random.uniform(-1e-4, 1e-4,
                               (self.state_width, self.input_width))
        b = np.zeros((self.state_width, 1))
        return Wh, Wx, b

    def forward(self, x):
        '''
        根据式1-式6进行前向计算
        '''
        self.times += 1
        # 遗忘门
        fg = self.calc_gate(x, self.Wfx, self.Wfh,
                            self.bf, self.gate_activator)
        self.f_list.append(fg)
        # 输入门
        ig = self.calc_gate(x, self.Wix, self.Wih,
                            self.bi, self.gate_activator)
        self.i_list.append(ig)
        # 输出门
        og = self.calc_gate(x, self.Wox, self.Woh,
                            self.bo, self.gate_activator)
        self.o_list.append(og)
        # 即时状态
        ct = self.calc_gate(x, self.Wcx, self.Wch,
                            self.bc, self.output_activator)
        self.ct_list.append(ct)
        # 单元状态
        c = fg * self.c_list[self.times - 1] + ig * ct
        self.c_list.append(c)  # *是对应元素的乘
        # 输出
        h = og * self.output_activator.forward(c)
        self.h_list.append(h)
        return h

    def calc_gate(self, x, Wx, Wh, b, activator): #函数内
        '''
        计算门
        '''
        h = self.h_list[self.times - 1]  # 上次的LSTM输出
        net = np.dot(Wh, h) + np.dot(Wx, x) + b
        gate = activator.forward(net)
        return gate

    def backward(self, x, delta_h):  #delta_h就是最后一个
        '''
        实现LSTM训练算法
        '''
        self.calc_delta(delta_h)
        self.calc_gradient(x)

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.Wfh -= self.learning_rate * self.Wfh_grad
        self.Wfx -= self.learning_rate * self.Wfx_grad
        self.bf -= self.learning_rate * self.bf_grad
        self.Wih -= self.learning_rate * self.Wih_grad
        self.Wix -= self.learning_rate * self.Wix_grad
        self.bi -= self.learning_rate * self.bi_grad
        self.Woh -= self.learning_rate * self.Woh_grad
        self.Wox -= self.learning_rate * self.Wox_grad
        self.bo -= self.learning_rate * self.bo_grad
        self.Wch -= self.learning_rate * self.Wch_grad
        self.Wcx -= self.learning_rate * self.Wcx_grad
        self.bc -= self.learning_rate * self.bc_grad

    def calc_delta(self, delta_h): #函数内
        # 初始化各个时刻的误差项
        self.delta_h_list = self.init_delta()  # 输出误差项
        self.delta_o_list = self.init_delta()  # 输出门误差项
        self.delta_i_list = self.init_delta()  # 输入门误差项
        self.delta_f_list = self.init_delta()  # 遗忘门误差项
        self.delta_ct_list = self.init_delta()  # 即时输出误差项

        # 保存从上一层传递下来的当前时刻的误差项，上一层什么意思？？？？？
        self.delta_h_list[-1] = delta_h

        # 迭代计算每个时刻的误差项
        for k in range(self.times, 0, -1):  #反向计算
            self.calc_delta_k(k)

    def init_delta(self):#函数内
        '''
        初始化误差项
        '''
        delta_list = []
        for i in range(self.times + 1):
            delta_list.append(np.zeros(
                (self.state_width, 1)))
        return delta_list

    def calc_delta_k(self, k):#函数内
        '''
        根据k时刻的delta_h，计算k时刻的delta_f、
        delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
        '''
        # 获得k时刻前向计算的值
        ig = self.i_list[k]
        og = self.o_list[k]
        fg = self.f_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k - 1]  #这要用到上一层的Ct-1
        tanh_c = self.output_activator.forward(c)
        delta_k = self.delta_h_list[k]  #取值这一层的dh（k）

        # 根据式9计算delta_o
        delta_o = (delta_k * tanh_c *
                   self.gate_activator.backward(og))
        delta_f = (delta_k * og *
                   (1 - tanh_c * tanh_c) * c_prev *
                   self.gate_activator.backward(fg))
        delta_i = (delta_k * og *
                   (1 - tanh_c * tanh_c) * ct *
                   self.gate_activator.backward(ig))
        delta_ct = (delta_k * og *
                    (1 - tanh_c * tanh_c) * ig *
                    self.output_activator.backward(ct))
        delta_h_prev = (
            np.dot(delta_o.transpose(), self.Woh) +
            np.dot(delta_i.transpose(), self.Wih) +
            np.dot(delta_f.transpose(), self.Wfh) +
            np.dot(delta_ct.transpose(), self.Wch)
        ).transpose()  #计算上一层的dh（k-1）

        # 保存全部delta值
        self.delta_h_list[k - 1] = delta_h_prev
        self.delta_f_list[k] = delta_f
        self.delta_i_list[k] = delta_i
        self.delta_o_list[k] = delta_o
        self.delta_ct_list[k] = delta_ct

    def calc_gradient(self, x):  #函数内
        # 初始化遗忘门权重梯度矩阵和偏置项
        self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
            self.init_weight_gradient_mat())
        # 初始化输入门权重梯度矩阵和偏置项
        self.Wih_grad, self.Wix_grad, self.bi_grad = (
            self.init_weight_gradient_mat())
        # 初始化输出门权重梯度矩阵和偏置项
        self.Woh_grad, self.Wox_grad, self.bo_grad = (
            self.init_weight_gradient_mat())
        # 初始化单元状态权重梯度矩阵和偏置项
        self.Wch_grad, self.Wcx_grad, self.bc_grad = (
            self.init_weight_gradient_mat())

        # 计算对上一次输出h的权重梯度
        for t in range(self.times, 0, -1):
            # 计算各个时刻的梯度  因为实际梯度是各个时刻的求和，因为这是一个样本，所以求和，若是多个样本还要求均值程序要改变
            (Wfh_grad, bf_grad,
             Wih_grad, bi_grad,
             Woh_grad, bo_grad,
             Wch_grad, bc_grad) = (
                self.calc_gradient_t(t))
            # 实际梯度是各时刻梯度之和
            self.Wfh_grad += Wfh_grad
            self.bf_grad += bf_grad
            self.Wih_grad += Wih_grad
            self.bi_grad += bi_grad
            self.Woh_grad += Woh_grad
            self.bo_grad += bo_grad
            self.Wch_grad += Wch_grad
            self.bc_grad += bc_grad

        # 计算对本次输入x的权重梯度
            #print('t:',t)
            xt = x[t-1].transpose()
            self.Wfx_grad += np.dot(self.delta_f_list[t], xt)
            self.Wix_grad += np.dot(self.delta_i_list[t], xt)
            self.Wox_grad += np.dot(self.delta_o_list[t], xt)
            self.Wcx_grad += np.dot(self.delta_ct_list[t], xt)

    def init_weight_gradient_mat(self):  #函数内
        '''
        初始化权重矩阵
        '''
        Wh_grad = np.zeros((self.state_width,
                            self.state_width))
        Wx_grad = np.zeros((self.state_width,
                            self.input_width))
        b_grad = np.zeros((self.state_width, 1))
        return Wh_grad, Wx_grad, b_grad

    def calc_gradient_t(self, t):  #函数内
        '''
        计算每个时刻t权重的梯度
        '''
        h_prev = self.h_list[t - 1].transpose()  #上一时刻的h（t-1）的转置,同self.h_list[t - 1].T
        Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
        bf_grad = self.delta_f_list[t]
        Wih_grad = np.dot(self.delta_i_list[t], h_prev)
        bi_grad = self.delta_f_list[t]
        Woh_grad = np.dot(self.delta_o_list[t], h_prev)
        bo_grad = self.delta_f_list[t]
        Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
        bc_grad = self.delta_ct_list[t]
        return Wfh_grad, bf_grad, Wih_grad, bi_grad, \
               Woh_grad, bo_grad, Wch_grad, bc_grad

    def reset_state(self):
        # 当前时刻初始化为t0
        self.times = 0
        # 各个时刻的单元状态向量c
        self.c_list = self.init_state_vec()
        # 各个时刻的输出向量h
        self.h_list = self.init_state_vec()
        # 各个时刻的遗忘门f
        self.f_list = self.init_state_vec()
        # 各个时刻的输入门i
        self.i_list = self.init_state_vec()
        # 各个时刻的输出门o
        self.o_list = self.init_state_vec()
        # 各个时刻的即时状态c~
        self.ct_list = self.init_state_vec()


def data_set(N):
    '''x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]  #data有两个值
    d = np.array([[1], [2]])'''
    batch_xs, batch_ys = mnist.train.next_batch(N)
    return batch_xs, batch_ys


def main():

    #训练
    time1=time.time()
    x, d = data_set(1000)  #10000
    lstm = LstmLayer(28, 100, 0.1)  #0.05,100样本不断循环可以学到
    weight=np.random.randn(10,100)
    bias=np.random.randn(10,1)
    for iteration in range(1000):  #zh,350,1000,0.6
        k=0

        for I in range(1000):  #1000个样本
            X=x[I].reshape([28,28,1])
            Y=d[I]
            for i in range(28):  #28步长
                input_t=X[i]
                h=lstm.forward(input_t)
            h=(np.dot(weight,h)+bias).reshape(10,)
            #print('h:', h)
            sensitivity_array = (h-Y).reshape(10,1)
            sensitivity_array=np.dot(weight.T,sensitivity_array)
            if np.argmax(h)==np.argmax(Y):
                k=k+1
            lstm.backward(X,sensitivity_array)
            lstm.update()
            lstm.reset_state()

        print('iteration:', iteration)
        accuracy=k/1000
        print('accuracy:',accuracy)
        if accuracy>=0.99:
            break

    #测试


    k = 0
    x, d = data_set(500)
    for I in range(500):  # 10个样本
        X = x[I].reshape([28, 28, 1])
        Y = d[I]
        for i in range(28):  # 28步长
            input_t = X[i]
            h = lstm.forward(input_t)
        h = (np.dot(weight, h) + bias).reshape(10, )
        if np.argmax(h) == np.argmax(Y):
            k = k + 1
        lstm.reset_state()
    print('测试accuracy:', k / 500)
    time2 = time.time()
    print('耗时：',time2-time1)

main()

