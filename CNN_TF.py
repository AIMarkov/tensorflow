import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define placeholder for inputs to network
# 定义数据
# 28x28,784是一张展平的图片，None表示输入图片的数量不定
xs = tf.placeholder(tf.float32, [None, 784])
print(xs)
# 图片类别(数字0-9)
ys = tf.placeholder(tf.float32, [None, 10])
print(ys)
# 定义后面dropout的占位符keep_prob
keep_prob = tf.placeholder(tf.float32)
# reshape将图片还原为28*28的
x_image = tf.reshape(xs, [-1, 28, 28, 1])


# 计算准确率
def compute_accuracy(v_xs, v_ys):
    global prediction
    # feed_dict是给placeholder创建出来的tensor赋值。
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # tf.equal(),比较两个矩阵的元素，相等则返回true，否则返回false
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # tf.cast(),将correct_prediction的数据类型转化成tf.float32
    # reduce_mean()求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 计算权值
# 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积操作
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling层输出值的计算
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])    # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
# 调用激活函数relu,即max(features,0)
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 28x28x32
# Pooling层输出值的计算，这里是在2x2的样本中取最大值
h_pool1 = max_pool_2x2(h_conv1)    # output size 14x14x32

# conv2 layer
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

# func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 构造阶段完成后，才能启动图
sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
