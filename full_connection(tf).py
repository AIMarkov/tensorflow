# *_* coding:utf-8 *_*

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data

# 返回字典,第一个参数是路径,one_hot=True表示长度为n的数组，只有一个元素是1.0
mnist = tf.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()

# 第一个None是表示批量的样本输入,x不是一个特定的值，而是一个占位符placeholder
x = tf.placeholder("float", shape=[None, 784])
# 第一个None表示批量的样本标签
y_ = tf.placeholder("float", shape=[None, 10])

#赋予tf.Variable不同的初值来创建不同的Variable
W1 = tf.Variable(tf.random_normal([784,200]))
W2 = tf.Variable(tf.random_normal([200,10]))
b1 = tf.Variable(tf.zeros([200]))
b2 = tf.Variable(tf.zeros([10]))

# 初始化变量
sess.run(tf.initialize_all_variables())

a1 = tf.sigmoid(tf.matmul(x, W1) + b1)
a2 = tf.matmul(a1,W2) + b2
y = tf.nn.softmax(a2)



# y_是标签给的，y是我们预测的
#首先，用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。
corss_entropy = -tf.reduce_sum(y_ * tf.log(y))

#自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(corss_entropy)

#使用with关键字的时候，就可以在Session中直接执行operation.run()或tensor.eval()两个类型的命令
with sess.as_default():
    for i in range(4000):
        batch = mnist.train.next_batch(50)
        #循环的每个步骤中，我们都会随机抓取训练数据中的50个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
        train_step.run(feed_dict={x: batch[0], y_: batch[1]},session=sess)

# correct_pre返回布尔值的list,argmax(y, 1)1是找的每一行的最大值下标，0是列,返回的都是List
correct_pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels},session=sess))
sess.close()

