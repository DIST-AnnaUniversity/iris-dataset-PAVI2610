#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# In[3]:


df = pd.read_csv('Iris.csv')
df.head()


# In[6]:


df = df.drop(['Id'],axis=1)
df.isnull().sum()


# In[30]:


import seaborn as sns
 
iris = sns.load_dataset('iris')
 
# style used as a theme of graph
# for example if we want black
# graph with grid then write "darkgrid"
sns.set_style("whitegrid")
 
# sepal_length, petal_length are iris
# feature data height used to define
# Height of graph whereas hue store the
# class of iris dataset.
sns.FacetGrid(iris, hue ="species",
              height = 6).map(plt.scatter,
                              'sepal_length',
                              'petal_length').add_legend()


# In[25]:


#Importing libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split

#Importing dataset
df = pd.read_csv('Iris.csv')
df.head()

#Renaming columns
df.columns = ['Id','SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

#Mapping output values to int
df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2})

#Printing out pandas dataframe
df

#Defining input and target variables for both training and testing
X = df.iloc[:100,[0,1,2,3]].values
y = df.iloc[:100,[4]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[27]:


#Importing libraries
import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2*np.random.random((4,1)) - 1
    def sigmoid(self, x):
        return 1 /(1+np.exp(-x))
    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output= self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error*self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("weights: ")
    print(neural_network.synaptic_weights)
training_inputs = X_train
training_outputs = y_train
neural_network.train(training_inputs, training_outputs, 1000)

#Showing Synaptic weights after training
print("weights after training: ")
print(neural_network.synaptic_weights)

#Deploying Neuron on training data
predicted = neural_network.think(X_test)

#Transforming results into Pandas Dataframe
predicted_df = pd.DataFrame({'Result': predicted[:, 0]})


# In[ ]:




