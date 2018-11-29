import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("Mnist_Data",one_hot=True)

lr=0.001  #0.001
training_iters=1000


n_input=28#输入的层数
n_steps=28#28长度

n_hidden=100#隐藏层的神经元(cell)个数

n_class=10#输出类别
batch_size=10  #输入样本批次的数目

x=tf.placeholder(tf.float32,[None,n_steps,n_input])  #(28,28)
y=tf.placeholder(tf.float32,[None,n_class])  #(10,)

weights={
    'in':tf.Variable(tf.random_normal([n_input,n_hidden])),  #生成(28,100)的正态随机数组，默认均值为0，标准差为1
    'out':tf.Variable(tf.random_normal([n_hidden,n_class]))  #(100,10)
}
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden])),  #常量tensor(100,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_class]))  #(10,)
}
#这里的weight和biases生成之后就不变了，这个主要是为了来改变最终维度，因为cell个数100，h个数也是100，不一定和y中的标签维度为10，因此需要单独一层来改变维度，同理输入时也是要改变的
def LSTM(X,weights,biases):

    X=tf.reshape(X,[-1,n_input])  #(-1,28),因为matmul是一个2维的乘法所以要把X转换为2维
    x_in=tf.matmul(X,weights['in'])+biases['in']  #-1表示自动匹配(?,28).(28,100)+(100)
    x_in=tf.reshape(x_in,[-1,n_steps,n_hidden])  #(-1,28,100)
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=0.1,state_is_tuple=True)  #创建cell，100个
    #n_hidden表示神经元的个数，forget_bias就是LSTM门的初始忘记系数，如果等于1，就是不会忘记任何信息。如果等于0，
    # 就都忘记。state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示


    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)  #batch_size=10,BasicLSTMCell类提供的一个初始化函数，生成全0的状态包括了主线的状态和分线的状态，c,h

    # final_state[0]是cell state
    # final_state[1]是hidden_state
    #outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,x_in,dtype=tf.float32,initial_state=None,time_major=False)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in,  initial_state=init_state,time_major=False)  #运行网络并输出的是每一步的output存入output中，而你若选择的是lstm则输出的状态（是最后的，不是每一步的）包含主线的分线的状态
    #time_major是指其时间步是主维度还是副维度，false就是x_in的副维度，true就是其主维度

    results=tf.matmul(final_state[1],weights['out'])+biases['out'] #(10,100).(100,10)计算最后结果final_state=(c,h),我们用h来计算，在这个例子中final_state[1]=outputs[-1],其他就不样了

    return results

pred=LSTM(x,weights,biases)  #输出的预测值
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))  #计算成本函数（先计算了logits和labels的交叉熵，再求求平均值，默认求所有元素均值

train_op=tf.train.AdamOptimizer(lr).minimize(cost)  #成本函数最小化为目标

correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))  #判断两个tensor是否相等，返回一个bool的tensor
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))  #计算准确率
#相比于基础SGD算法，1.不容易陷于局部优点 2.速度更快


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #直接初始化

    for step in range(training_iters):

        #print('in:', final_state)  #final_state是一个LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_2:0' shape=(10, 100) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_3:0' shape=(10, 100) dtype=float32>)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #提取minist的下一个batch
        #print(batch_ys.shape) #(10,784)每次取10张图

        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])  #(10,28,28)
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})


        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        if step % 10 == 0:
            print("Iter " + str(step) + ", Testing Accuracy=" + str(acc))
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
    sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
    print("acuuracy:",str(acc))

