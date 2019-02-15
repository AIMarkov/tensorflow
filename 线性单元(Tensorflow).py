import tensorflow as tf
import numpy as np


x_data =np.asarray([ [0.29542801],[0.27334914],[0.48793942],[0.34162864],[0.68219209],[0.60673505],
           [0.40210965],[0.23888192],[0.948605],[0.0495657],[0.57764995],[0.83816797],
           [0.48592442],[0.51871407],[0.73516047],[0.69514716],[0.283856],[0.55346227]]).astype(np.float32)  #产生一个随机数组大小为(1,100)
y_data = x_data * 0.1 + 0.3 #由x_data生成了y_data所以他们的形式是一样的，所以不会有错
print(y_data)

W = tf.Variable(tf.random_uniform([1,1], 0, 0))  #随机生成服从均匀分布的一个Tensor对象大小[1],范围（-1,1）
b = tf.Variable(tf.zeros([1]))  #产生一个Tensor对象大小为[1],值为0
y = tf.matmul(x_data, W) + b  #y是线性回归运算产生的Tensor

loss = tf.reduce_mean(tf.square(y - y_data))  #定义损失
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)
print(sess.run(train))
# Fit the line.
for step in range(1000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

