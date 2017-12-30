
# coding: utf-8

# In[3]:

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D,Flatten


from keras.datasets import mnist




# In[4]:

# 获取数据集
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("../assert/data/", one_hot=True)  

x_train, y_train = mnist.train.images,mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.test.labels  
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')  
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')  


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_train, num_classes)

x_train.shape


# In[ ]:

# 设置训练参数，
learning_rate = 0.001
num_steps = 10
batch_size = 128
display_step = 10

# CNN参数
num_input = 784
num_classes = 10
dropout = 0.5


# In[ ]:

model = Sequential()

model.add(Conv2D(32,(5,5),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x_train,y_train,batch_size=batch_size, epochs=num_steps)

score = model.predict(x_test,y_test, batch_size=batch_size)

print("Test Score ", score)


# In[ ]:




