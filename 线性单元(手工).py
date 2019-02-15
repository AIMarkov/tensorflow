from functools import reduce
import math
class Perceptron(object):
    def __init__(self, input_num, activator):  # 1.初始化函数2.预测函数3.多轮训练函数4.更新函数5单轮训练函数
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator  # 这里输入的是感知器的激活函数
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]  # 权重都初始画为0,input_num=2,权重值也是list[0.0,0.0]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):  # 这个函数是被print()调用的
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,          # reduce的第一个参数为求和函数，第二个参数是map函数处理后的新list
                   list(map(lambda x, w: x * w,    # map的第一个参数是两个数的乘积
                       input_vec, self.weights))     # map的第二个参数是输入向量和权值向量打包后的list
                , 0.0) + self.bias) #最后返回的是调用了激活函数的返回值1或0

    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = list(zip(input_vecs, labels))
        loss=0
        i=0
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:  # 此时input_vec是一个样本，label是其标签如：[1, 1] 1
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)  # 返回预测标签
            # 更新权重
            loss+=self._update_weights(input_vec, output, label, rate)  # 输入此时的样本，预测值，真实标签，学习率
            i=i+1
        print('loss:',loss/i)
    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        #print(delta)
        self.weights = list(map(
            lambda x, w: w + rate * delta * x,
            input_vec, self.weights))
        # 更新bias
        self.bias += rate * delta
        return(delta**2)



class LinearUnit(Perceptron):  # 继承了Perceptronf
    def __init__(self, input_num,func):
        '''初始化线性单元，设置输入参数的个数'''
        self.func=func
        Perceptron.__init__(self, input_num, self.func)  # f定义在前面可以直接写？原来代码跟我不一样，看一下原来代码思路
def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限,第二项是职位1，2，3
    input_vecs =[ [0.29542801],[0.27334914],[0.48793942],[0.34162864],[0.68219209],[0.60673505],
           [0.40210965],[0.23888192],[0.948605],[0.0495657],[0.57764995],[0.83816797],
           [0.48592442],[0.51871407],[0.73516047],[0.69514716],[0.283856],[0.55346227]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [0.32954282, 0.32733494,0.34879395,0.33416289,0.36821923,0.36067352,0.34021097,0.32388821,0.39486051,0.30495659,0.35776502,0.38381681,0.34859246,0.35187143,0.37351605,0.36951473,0.32838562,0.35534623]
    return input_vecs, labels
def f(x):
    return x
def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(2,f)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 1000, 0.5)
    #返回训练好的线性单元
    return lu
if __name__ == '__main__':
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('[0.29542801]测试值：{}'.format(linear_unit.predict([0.29542801])))
    print('[0.48793942]测试值：{}'.format(linear_unit.predict([0.48793942])))
    print('[0.27334914]测试值：{}'.format(linear_unit.predict([0.27334914])))
    print('[0.34162864]测试值：{}'.format(linear_unit.predict([0.34162864])))