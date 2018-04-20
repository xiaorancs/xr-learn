# _*_ coding:utf-8 _*_

from __future__ import  division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 得到数据集合MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 设置全局变量的参数
learning_rate = 0.01

# 训练的步数和批次
num_steps = 3000
batch_size = 128

# 显示步数
display_step = 500
examples_to_show = 10

# 设置网络参数
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# 设置输入占位符
X = tf.placeholder("float", [None, num_input])

# 设置网络层的参数
weights = {
    "encoder_h1":tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    "encoder_h2":tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    "decoder_h1":tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    "decoder_h2":tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    "encoder_b1":tf.Variable(tf.random_normal([num_hidden_1])),
    "encoder_b2":tf.Variable(tf.random_normal([num_hidden_2])),
    "decoder_b1":tf.Variable(tf.random_normal([num_hidden_1])),
    "decoder_b2":tf.Variable(tf.random_normal([num_input])),
}

# build解码和编码层,使用sigmoid的激活函数
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2
# 构造model
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


# 定于损失和优化器
y_pre = decoder_op
y = X

# 使用平方误差进行优化
loss = tf.reduce_sum(tf.pow(y-y_pre,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

# 初始化全局变量
init = tf.global_variables_initializer()

# run netmodel, start train
sess = tf.Session()
sess.run(init)

# train
for i in range(1, num_steps+1):
    batch_x, _ = mnist.train.next_batch(batch_size=batch_size)
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    if i % display_step == 0 or i == 1:
        print ("Step %d: Minimie loss %f" % (i, l))

# test autoencode, 可视化对应的图片
n = 4
canvas_origin = np.empty((28 * n, 28 * n))
canvas_decoder = np.empty((28 * n, 28 * n))
for i in range(n):
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # 传入解码器，编码之后继续解码这个图片，得到对应的一批图片的解码
    g = sess.run(decoder_op, feed_dict={X: batch_x})
    
    # 显示原始的图片
    for j in range(n):
        # Draw the generated digits
        canvas_origin[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    # 显示重新构造的函数
    for j in range(n):
        # Draw the generated digits
        canvas_decoder[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

print("Original Images")     
plt.figure(figsize=(n, n))
plt.imshow(canvas_origin, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_decoder, origin="upper", cmap="gray")
plt.show()








