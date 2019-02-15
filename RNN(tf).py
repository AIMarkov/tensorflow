import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("Mnist_Data",one_hot=True)
lr = 0.001
training_iters=2000
n_input=28  #输入的层数
n_steps=28  #28长度
n_hidden=100#隐藏层的神经元个数
n_class=10#输出类别
batch_size=100
x=tf.placeholder(tf.float32,[None,n_steps,n_input])
y=tf.placeholder(tf.float32,[None,n_class])
weights={
    'in':tf.Variable(tf.random_normal([n_input,n_hidden])),
    'out':tf.Variable(tf.random_normal([n_hidden,n_class]))
}
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_class]))
}
def RNN(X,weights,biases):
    X=tf.reshape(X,[-1,n_input])
    x_in=tf.matmul(X,weights['in'])+biases['in']
    x_in=tf.reshape(x_in,[-1,n_steps,n_hidden])
    #lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=0.1,state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,x_in,dtype=tf.float32,initial_state=init_state,time_major=False)
    #outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in,  initial_state=init_state,time_major=False)
    #results=tf.matmul(final_state[1],weights['out'])+biases['out']
    results = tf.matmul(final_state, weights['out']) + biases['out']
    return results
pred=RNN(x,weights,biases)
#cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
cost=tf.reduce_mean(tf.square(y-pred))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_iters):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs = batch_xs.reshape([100, n_steps, n_input])
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
        #if step % 10 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        if step % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            print("Iter " + str(step) + ", Testing Accuracy=" + str(acc))

    for i in range(100):
        batch_xs, batch_ys = mnist.test.next_batch(100)
        batch_xs = batch_xs.reshape([100, n_steps, n_input])
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        print("Testing Iter " + str(i) + ", Testing Accuracy=" + str(acc))




