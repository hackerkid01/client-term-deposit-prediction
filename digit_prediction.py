#!/usr/bin/env python
# coding: utf-8

# ## Project name: Digits prediction using Street View Housing Number(SVHN) Dataset
# 


# ### Problem Statement:
# 
# - We have to build a predictive model which shall predict the digit corresponding to the 32*32 RGB image of SVHN dataset
# 
# 

# ### Dataset description:
# 
#     There are total 99289 instances in the dataset out of which 73257 instances belong to training data and 26032 instances belong to test data.
#     Each instance3 is a 32*32*3 RGB image.
# 
# One can find more about the dataset [here](http://ufldl.stanford.edu/housenumbers/).
# 

# ### Steps of our project:
# 
# - [Importing the dataset and understanding the dataset](#Import-the-dataset-and-understanding-the-dataset) 
# - [Preprocessing the data](#Preprocessing-the-data)
# - [Building the model](#Building-the-model)
# - [Training the Model](#Evaluation-of-the-Model)
# - [Trying out different optimizers](#Trying-out-different-optimizers)
# - [Making the predictions](#Making-the-predictions)
# - [Measuring the performance of the model](#Measuring-the-performance-of-the-model)

# ### Importing the dataset and understanding the dataset

# In[1]:


import scipy.io
train_data=scipy.io.loadmat('C:\\Users\\sshar\\Downloads\\train_32x32.mat')
test_data=scipy.io.loadmat('C:\\Users\\sshar\\Downloads\\test_32x32.mat')


# In[2]:


train_X = train_data['X'].copy()
train_y = train_data['y'].copy()
test_X=test_data['X'].copy()
test_y = test_data['y'].copy()


# In[3]:


print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)


# In[4]:


train_X[:,:,:,0]


# In[5]:


type(test_X)


# In[7]:


import matplotlib.pyplot as plt


plt.subplot(121)
plt.imshow(train_X[:,:,:,108])

plt.subplot(122)
plt.imshow(train_X[:,:,:,0])
train_y[0]


# In[8]:


print(train_y[1])


# In[9]:


train_y[1]


# In[10]:


train_X.shape


# In[11]:


import numpy as np
train_X=np.transpose(train_X,[3,0,1,2])
test_X=np.transpose(test_X,[3,0,1,2])


# In[12]:


test_X.shape


# In[13]:


train_y.shape


# In[14]:


for i in range(10):
    print(str(train_y[i]))


# # Preprocessing the data

# In[15]:


#normalize

train_X = train_X/255
test_X=test_X/255


# In[16]:


#one hot encoding of target values

from keras.utils import to_categorical

y_train_one_hot = to_categorical(train_y,num_classes=11)
y_test_one_hot = to_categorical(test_y,num_classes=11)


# In[17]:


y_test_one_hot[222]


# # Building the model

# We have used 3 CNN layers followed by relu and maxpooling layers after every CNN layer.
# 
# #### 1.1st CNN layer(8 filters each of size 5*5)
# #### 2.2nd CNN layer(16 filters each of size 3*3)
# #### 3.3rd CNN layer(32 filters each of size 3*3)
# 
# The input image is zero padded appropriately such that the length and breadth dimensions of the output image(obtained after each CNN layer) are same to that of the input image.
# The depth of the output image is same as the number of filters applied in the corresponding CNN layer.
# 
# Maxpooling layer reduces the spatial size of the image.If the image has a spatial dimension of nxn then after maxpooling, the spatial size reduces to (n/2)x(n/2).
# 
# #### Significance of the maxpooling layers
# 
# Due to reduction in the spatial size of the image,the amount of parameters and computation in the network reduces which in turn helps to control overfitting.
# 
# #### In the ANN part,we have
# 
# 1.input layer(512 nodes)<br>
# 2.one hidden layer(256 nodes)<br>
# 3.output layer(11 nodes)<br><br>
# (As per the SVHN data set 0th node is of no significance,nodes numbered 1 to 9 correspond to digits 1 to 9 respectively and the 10th node corresponds to digit 0.
# Hence there are 11 nodes numbered 0 to 10 in the output layer)

# In[18]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input

row = 32
col = 32
n_chnl = 3

my_cnn = Sequential()

# Conv2D (no_of_filters, (filter row, filter column), stride = 1, activation = 'relu')
my_cnn.add(Conv2D(16, (5,5), activation = 'relu', padding='same', input_shape = (row,col,n_chnl)))

# MaxPool2D (pool_size = (2,2))
my_cnn.add(MaxPool2D((2,2)))

my_cnn.add(Conv2D(32, (3,3), activation = 'relu', padding='same'))
my_cnn.add(MaxPool2D((2,2)))

my_cnn.add(Conv2D(64, (3,3), activation = 'relu', padding='same'))
my_cnn.add(MaxPool2D((2,2)))

my_cnn.add(Flatten())
my_cnn.add(Dense(256,activation = 'relu'))
my_cnn.add(Dense(11, activation='softmax'))


# ## Trying out different optimizers
# 
# Keeping the numbers and dimensions of the filters same in the different layers and loss as  'categorical_creosssentropy' ,we tried out different optimizers and the got the following accuracies respectively -
# 
# #### 1.adam (89.01 %)
# #### 2.Stocastic gradient descent (89.44 %)
# #### 3.RMSprop (8.16 %)
# #### 4.Adagrad (88.88 %)
# #### 5.Adadelta (88.18 %)
# #### 6.Nadam (87.84 %)
# #### 7.Adamax(89.6 %)
# 
# #### Since we obtained maximum accuracy for 'Adamax' optimizer,we have shown the further computations using the same. 
# 
# 

# In[61]:


my_cnn.compile(loss='categorical_crossentropy',optimizer='Adadelta', metrics=['accuracy'])


# In[62]:


my_cnn.summary()


# # Training the model

# In[63]:


batch_size=128
n_epochs=10

history = my_cnn.fit(train_X, y_train_one_hot, batch_size=batch_size, epochs=n_epochs, shuffle=True)


# # Making the predictions

# In[64]:


y_pred = my_cnn.predict_classes(test_X)

y_pred.shape


# # Measuring the performance of the model

# In[65]:


from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_pred = y_pred, y_true=test_y)
conf_mat


# In[66]:


accuracy = 100*np.trace(conf_mat)/np.sum(conf_mat)
accuracy


# In[67]:


print("Accuracy of the model is {}".format(accuracy))


# In[ ]:





# In[ ]:




