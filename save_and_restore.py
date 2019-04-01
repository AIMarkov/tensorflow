
import tensorflow as tf
import numpy as np

X=tf.placeholder(dtype=tf.float32,shape=[5,1])
Y=tf.placeholder(dtype=tf.float32,shape=[5,1])

W=tf.Variable(tf.zeros([1,1]))
b=tf.Variable(tf.ones([1]))

out=tf.matmul(X,W)+b

loss=tf.reduce_mean(tf.square(out-Y))
train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess=tf.Session()
init=tf.global_variables_initializer()

sess.run(init)

saver=tf.train.Saver(max_to_keep=0)
for i in range(10000):
    x = np.random.uniform(0, 10000, [5, 1])  # 这个值要小一点才好,太大了要发散(梯度太大了),所以进行预处理
    x_mean=np.mean(x)
    x =x/x_mean
    y = 2 * x + 3
    sess.run(train,feed_dict={X:x,Y:y})
    #print(sess.run(loss,feed_dict={X:x,Y:y}))
    #print(sess.run(W),sess.run(b))
    saver.save(sess, 'ckpt/model.ckpt')
#test
model_file=tf.train.latest_checkpoint('ckpt/')
saver.restore(sess,model_file)
x = np.random.uniform(0, 10000, [5, 1])

y = 2 * x + 3
print(y)
print(sess.run(out,feed_dict={X:x,Y:y}))
