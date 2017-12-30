
# coding: utf-8

# # 基于tensorflow的CNN实现
# 

# In[1]:

import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data



# In[2]:

mnist = input_data.read_data_sets("../assert/data/",one_hot=True)


# In[3]:

# 设置训练参数，
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# CNN参数
num_input = 784
num_classes = 10
dropout = 0.5


# In[4]:

# 1. 定义输入输出占位符
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

# dropout的使用
keep_prob = tf.placeholder(tf.float32)


# In[5]:

# 2. 创建简单的卷积接口
def conv2d(x,W,b,strides = 1):
    # relu激活函数，滑动窗口是1
    x = tf.nn.conv2d(x, W, strides=[1,strides,strides, 1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    
    return tf.nn.relu(x)

# 池化接口
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1],padding='SAME')


# create model
def conv_net(x, weights, biases, dropout):
    # MNIST 输入是784维 reshape to 高 x 宽 x 通道数
    # Tensor input bcome 4-d [batch_size, hight, width, channel]
    x = tf.reshape(x, shape=[-1,28,28,1])
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    
    # 全链接层
    fc1 = tf.reshape(conv2,[-1, 7*7*64])
#     print(fc1.get_shape())
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    
    return out


# In[ ]:




# In[6]:

# 设置变量
weights = {
    # 5x5 conv ,1 input 32 outputs
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    
    # 5x5 conv ,32 input 64 outputs
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    
    # fully connected,7*7*64 input2, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64,1024])),
    
    # 1024 input, 10 outputs
    'out': tf.Variable(tf.random_normal([1024,num_classes]))
    
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes])),
    
}



# In[11]:

# 构造网络
logits = conv_net(X, weights,biases, keep_prob)
prediction = tf.nn.softmax(logits)

# loss 和 optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y ,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# In[ ]:

# start training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(1,num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        
        if step == 1 or step % display_step == 0:
            loss, acc = sess.run([loss_op,accuracy], feed_dict={X: batch_x, Y:batch_y, keep_prob: 1.0})
            
            print("Step "+str(step)+",Minibatch Loss = "+                  "{:.4f}".format(loss)+", Training Accuracy= "+                  "{:.3f}".format(acc))
    print("Optinization Finished!")
    
    
    # test
    print("Test accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images[:256],
                                                          Y:mnist.test.labels[:256],
                                                          keep_prob: 1.0}))
    


# In[ ]:



