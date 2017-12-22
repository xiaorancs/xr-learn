
# coding: utf-8

# In[1]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import Adam

from keras.datasets import mnist


# In[30]:

# step.1 gorbal parameters
learning_rate = 0.1
num_steps = 10
batch_size = 128
dispaly_step = 100
num_classes = 10

dim = 28*28


# In[33]:

# 需要翻墙,下载失败
# (x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[39]:

# 获取数据集
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("../assert/data/", one_hot=True)  

x_train, y_train = mnist.train.images,mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.test.labels  
x_train = x_train.reshape(-1, 28 * 28).astype('float32')  
x_test = x_test.reshape(-1, 28 * 28).astype('float32')  


# In[40]:

y_test.shape


# In[41]:

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_train, num_classes)


# In[ ]:




# In[42]:

model = Sequential()

# 网络结构
model.add(Dense(256, activation='relu',input_dim = dim))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=num_steps,batch_size=batch_size)

# 验证得分
score = model.evaluate(x_test,y_test,batch_size=batch_size)

print("score = ", score)


# In[43]:




# In[ ]:



