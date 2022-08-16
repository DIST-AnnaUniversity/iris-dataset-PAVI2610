#!/usr/bin/env python
# coding: utf-8

# In[23]:


#import the libraries
import pandas as pd
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
f=pd.read_csv("F:\dataset/iris_data.csv")
print(f)


# In[24]:


#input
x=f.iloc[:,0:4].values
x


# In[27]:


y1=f.iloc[:,4].values
y1


# In[25]:


y1=f.iloc[:,4].values
y1
y=np.where(y1=='Iris-setosa',1,-1)
y


# In[3]:


import numpy as np
import pandas as pd
f=pd.read_csv("F:\dataset/iris_data.csv")
print(f)
#/----------------Function for perceptron algorithm --------------/
def perceptron(c,X,d,w,iteration):
    for n in range(1,iteration):# Number of iterations  = 7
        print("Iteration :",n)
        for i, x in enumerate(X):
            net = np.dot(X[i],w)
            if net > 0:
                out = 1
            else:
                out = -1
            r = c*(d[i] - out)
            delta_w = r*x
            w = delta_w+w
            print (n, i, w)
    return w

#---------------Training---------------------------------/
X =f.iloc[:,0:4].values
X
d1=f.iloc[:,4].values
d=np.where(d1=='Iris-setosa',1,-1)
d
w= ([0,0,0,0])
print ("initial values of weights", w)
c =1
iterations=100
print ("Training")
print ("----------")
final_weight = perceptron(c,X,d,w,iterations)
print ("Final sets of weights: ", final_weight)


# In[4]:


import numpy as np

import pandas as pd
f=pd.read_csv("F:\dataset/iris_data.csv")
from sklearn.model_selection import train_test_split
#/----------------Function for perceptron algorithm --------------/
def perceptron(c,X,d,w,iteration):
    for n in range(1,iteration):# Number of iterations  = 10
        print("Iteration :",n)
        for i, x in enumerate(X):
            net = np.dot(X[i],w)
            if net > 0:
                out = 1
            else:
                out = -1
            r = c*(d[i] - out)
            delta_w = r*x
            w = delta_w+w
            print (n, i, w)
    return w
#/---------------------Function for testing the perceptron-----------/

def test_perceptron(final_out,X,w):
    for i,x in enumerate(X):
        net = np.dot(X[i],w)
        if net>0:
            out = 1
        else:
            out = -1
        final_out = final_out+[out]
    return final_out

#---------------Training---------------------------------/
X = f.iloc[:,0:4].values
print ("Inputs", X)
d1=f.iloc[:,4].values
d=np.where(d1=='Iris-setosa',1,-1)
print ("Teacher values", d)
#Splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
w= ([0,0,0,0])
print ("initial values of weights", w)
c = 1
iterations = 25
print ("Training")
print ("----------")
final_weight = perceptron(c,X_train,y_train,w,iterations)
print ("Final sets of weights: ", final_weight)
#-----------------Testing-------------------------------/
final_out = []
print ("Testing")
print ("--------")
final_output = test_perceptron(final_out,X_test,final_weight)
print ("Final output: ", final_output)
print ("Original Teacher values",y_test)


# In[ ]:




