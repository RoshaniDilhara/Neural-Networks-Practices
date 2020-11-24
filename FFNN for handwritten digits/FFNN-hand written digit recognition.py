#!/usr/bin/env python
# coding: utf-8

# # Identify hand written digits using MNIST dataset using FFNN

# In[2]:


from keras.datasets import mnist

(train_data,train_target),(test_data,test_target) = mnist.load_data()


# In[3]:


print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)


# In[10]:


import matplotlib.pyplot as plt

plt.imshow(train_data[0],cmap='gray')
plt.show()


# In[13]:


train_target[0]


# In[15]:


from keras.model import Sequential
from keras.layers import Dense,Flatten

model = Sequential() #creating a new sequaentail type empty NN model

#we cant directly apply the 2D image to te NN -> so have to flatten the image-> 1x (28x28)
#convert 2D image to flatten layer
#784 neurons for the first layer
#this is the first inpiut layer
model.add(Flatten(input_shape=(28,28)))

#this is a classification problem 
model.add(Dense(512,activate='relu'))
model.add(Dense(256,activate='relu'))
model.add(Dense(128,activate='relu'))
model.add(Dense(64,activate='relu'))
model.add(Dense(10,activate='softmax')) #here we have 10 types of numbers: 0-9

#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


#categorical conversion

from keras.utils import np_utils

#train_target,test_target will be converted to categoical form
new_train_target = np_utils.to_categorical(train_target)
new_test_target = np_utils.to_categorical(test_target)

print(train_target[:10])
print(new_train_target[:10])


# In[ ]:


#each pixel contain a value between 0-255
#usually we are not gonna apply these big values to NNs
#We always try to normalize it and scale down 
# to scale down-> devide the data from 255


# In[ ]:


# all the pixels will be devided by 255
new_train_data = train_data/255
new_test_data = test_data/255

# this converts the pixels which has the range of 0-255 into 0-1


# In[ ]:


model.fit(new_train_data,new_test_data,epochs=20)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(model.history.history['loss'])
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.show()


# In[ ]:


plt.plot(model.history.history['accuracy'])
plt.xlabel('# epochs')
plt.ylabel('accuracy')
plt.show()


# In[ ]:


model.evaluate(new_test_data,new_test_target)


# In[ ]:


# after training a neural network, we can save it to a physical file(the knowledge gained by the dataset)
# then that can we used in other applications

#saving the weights of the knowledge gained by the NN from the trained set
model.save_weights('FFNN-MNIST.h5')


# In[ ]:


# Can visualize the trained weights and biasesof the NN

for layer in model.layers:
    
    parameters = layer.get_weights() #get weight of each layer
    #there are two elements in this parameter: weights and bias
    weights = parameters[0]
    biases = parameters[1]
    print('weights:', weights)
    print('biases:', biases)
    print('==================================================================')

