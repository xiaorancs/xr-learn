
# coding: utf-8

# In[1]:


# Author: xiaoran
# Time: 2017-12-21 21:00
# 基于tensorflow的神经网路


# In[2]:

from __future__ import print_function

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# read data, one-hot encoding
mnist = input_data.read_data_sets("../assert/data/",one_hot=True)

import tensorflow as tf


# In[3]:

# step.1 gorbal parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
dispaly_step = 100

# NetWork parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2en layer number of neurons
num_input = 784 # MNIST data input (28*28)
num_classes = 10 # MNIST data classes(0-9)


# In[4]:

# Set input placeholder
X = tf.placeholder("float",[None, num_input])
Y = tf.placeholder("float",[None, num_classes])

# set layer weight & bias, variable
weights = {
    'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([num_classes])) 
}

# create model struct of network
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
    
    out_layer = tf.add(tf.matmul(layer_2, weights['out']),biases['out'])
    
    return out_layer


# In[6]:

# construct model
y_ = neural_net(X)

# define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Evaluate mmodel with test 
correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# In[10]:

# start train
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(1,num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        sess.run(train, feed_dict={X: batch_x, Y:batch_y})
        if step % dispaly_step == 0 or step == 1:
            losses, acc = sess.run([loss, accuracy], feed_dict={X: batch_x,
                                                                Y: batch_y})
            print("Step "+str(step)+", Minibatch Loss = " + "{:.4f}".format(losses)                  + ", Training Accuracy= " + "{:.3f}".format(acc) )
    print("Finished!")
    
    print("Testing Accuracy: ",          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                       Y: mnist.test.labels}))


# In[ ]:



