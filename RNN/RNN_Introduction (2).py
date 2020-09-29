#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy


# In[2]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivativeSigmoid(x):
    return x * (1 - x)


# In[3]:


binary_dim = 8
largest_num = pow(2, binary_dim)
int2bin = {}


# In[4]:


largest_num


# In[5]:


binary = np.unpackbits(np.array([range(largest_num)], dtype=np.uint8).T, axis = 1)


# In[6]:


binary


# In[7]:


binary.shape


# In[8]:


binary[5]


# In[9]:


for i in range(largest_num):
    int2bin[i] = binary[i]


# In[ ]:





# In[10]:


alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


# In[11]:


wh = np.random.random((input_dim, hidden_dim))
wout = np.random.random((hidden_dim, output_dim))
wtime = np.random.random((hidden_dim, hidden_dim)) 


# In[12]:


wh_update = np.zeros_like(wh)
wout_update = np.zeros_like(wout)
wtime_update = np.zeros_like(wtime)


# In[13]:


a_int = np.random.randint(largest_num / 2)
b_int = np.random.randint(largest_num / 2)
a = int2bin[a_int]
b = int2bin[b_int]

c_int = a_int + b_int
c = int2bin[c_int]

d = np.zeros_like(c)
overallError = 0

layer_1_values = []
layer_1_values.append(np.zeros(hidden_dim))

layer_2_deltas = []


# In[14]:


layer_1_values


# In[15]:


X = np.array([[a[binary_dim - 0 - 1], b[binary_dim - 0 - 1]]])
y = np.array([[c[binary_dim - 0 - 1]]])


# In[16]:


X


# In[17]:


y


# In[18]:


layer_1 = sigmoid(np.dot(X, wh) + np.dot(layer_1_values[-1], wtime))


# In[19]:


layer_1


# In[20]:


layer_2 = sigmoid(np.dot(layer_1, wout))


# In[21]:


layer_2


# In[22]:


output_error = y - layer_2


# In[23]:


output_error


# In[24]:


output_slope = derivativeSigmoid(layer_2[0][0])


# In[25]:


output_slope


# In[26]:


layer_2_deltas.append(output_error * output_slope)


# In[27]:


layer_2_deltas


# In[28]:


overallError += np.abs(output_error[0])


# In[29]:


overallError


# In[30]:


d


# In[31]:


d[binary_dim - 0 - 1] = np.round(layer_2[0][0])


# In[32]:


d


# In[33]:


epochs = 100000

for epoch in range(epochs):
    a_int = np.random.randint(largest_num / 2)
    b_int = np.random.randint(largest_num / 2)
    a = int2bin[a_int]
    b = int2bin[b_int]
    
    c_int = a_int + b_int
    c = int2bin[c_int]
    
    d = np.zeros_like(c)
    overallError = 0
    
    layer_1_values = []
    layer_1_values.append(np.zeros(hidden_dim))
    
    layer_2_deltas = []
    
    for position in range(binary_dim):
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]])
        
        # hidden layer
        layer_1 = sigmoid(np.dot(X, wh) + np.dot(layer_1_values[-1], wtime))
        # output layer
        layer_2 = sigmoid(np.dot(layer_1, wout))
        
        output_error = y - layer_2
        output_slope = derivativeSigmoid(layer_2[0][0])
        layer_2_deltas.append(output_error * output_slope)
        
        overallError += np.abs(output_error[0])
        
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        layer_1_values.append(copy.deepcopy(layer_1))
        
    future_layer_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]
        
        delta_output = layer_2_deltas[-position - 1]
        
        hidden_error = future_layer_delta.dot(wtime.T) + delta_output.dot(wout.T)
        slope_hidden = derivativeSigmoid(layer_1)
        hidden_delta = hidden_error * slope_hidden
        
        wout_update += np.atleast_2d(layer_1).T.dot(delta_output)
        wtime_update += np.atleast_2d(prev_layer_1).T.dot(hidden_delta)
        wh_update += X.T.dot(hidden_delta)

        future_layer_delta = hidden_delta
        
    wh += wh_update * alpha
    wtime += wtime_update * alpha
    wout += wout_update * alpha
    
    wh_update *= 0
    wtime_update *= 0
    wout_update *= 0
    
    if(epoch % 1000 == 0):
        print("Error : {}".format(overallError))
        print("Pred : {}".format(d))
        print("Actual : {}".format(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x * pow(2, index)
            
        print("Actual -> {} + {} = {}".format(a_int, b_int, c_int))
        print("Predicted -> {} + {} = {}".format(a_int, b_int, out))
        print("*" * 20)


# In[ ]:





# In[ ]:





# In[ ]:




