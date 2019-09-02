# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:52:45 2018

@author: craig
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:55:33 2018

@author: craig
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tensorflow as tf
import time 

t0 = time.clock()

#結構初始化
tf.reset_default_graph()

def pre_dis_cal(TM_X, TM_Y):
    distance = np.zeros((12,12))
    for row in range(0, 12):
        for column in range(row, 12):
            distance[row, column] = np.sqrt((TM_X[row] - TM_X[column])**2 + (TM_Y[row] - TM_Y[column])**2)  
    for row in range(0, 12):
        for column in range(row, 12):
             distance[column, row] = distance[row, column]
    return distance
            
def dis_cal(X, Y, TM_X, TM_Y):
    distance = np.zeros((1,12))
    for column in range(0, 12):
        distance[0, column] = np.sqrt((X - TM_X[column])**2 + (Y - TM_Y[column])**2)  
    return distance

def make_matrix(Z, num_data, TM_X, TM_Y):
    matrix = np.zeros([180,180])
    for a in range(num_data):
        row = int(TM_X[a])-2030
        column = int(TM_Y[a])-2630
        matrix[row, column] = Z[i][a]
    return matrix

Data = pd.read_excel("地下水位(日)_台中地區_測試.xlsx", header=None)
Data = np.array(Data)
TM_X = Data[0,4:-2] / 100
TM_Y = Data[1,4:-2] / 1000
num_data = len(TM_X)
Z = np.zeros((len(TM_X)))
for i in range(31):
    Z =  np.vstack((Z, Data[1464+i,4:-2]))
Z = Z[1:,:]

matrix = make_matrix(Z, num_data, TM_X, TM_Y)
train_dis = pre_dis_cal(TM_X, TM_Y)

temp1 = np.zeros((1,1))
temp2 = np.zeros((1,1))
for i in range(12):
   temp1 = np.vstack((temp1, train_dis[i].reshape(-1,1)))
   temp2 = np.vstack((temp2, Z[0].reshape(-1,1)))
Train_X = np.hstack((temp1[1:,:], temp2[1:,:]))
Train_Y = Z[0].reshape(-1,1)

#------------------------------------------------------RNN--------------------------------------------------------

#設定及初始化參數
learning_rate = 0.0005
epoches = 10000
input_size = 2
n_steps = 12
batch_size = 1
output_size = 1
W = []
b = []

x = tf.placeholder(tf.float32, [None, n_steps, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

#設定各層神經元數量
n_hidden_units = 300
n_neurons = [input_size, n_hidden_units, output_size]

#建構權重及偏差
for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))

#RNN 模型建構
def RNN(X, W, b):
    #reshape X to 2-dimensional to compute
    X = tf.reshape(X, [-1, input_size])
    X_in = tf.matmul(X, W[0]) + b[0]
    
    #reshape X to 3-dimensional to put in dynamic_rnn
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    #creat a lstm cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    #tf.nn.dynamic_rnn expects a 3-dimensional tensor as input
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], W[-1]) + b[-1]  
    
    return results

#定義損失函數，此為平方差代價函數
def Cost(y_label, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))
    return square_error

prediction = RNN(x, W, b)

cost = Cost(y, prediction)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#---------------------------train---------------------------

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

costs = np.ones((epoches,1))

for i in range(epoches):
    num = (i * 12) % (144-12)
    num2 = i % 12
    sess.run(train, feed_dict={x: Train_X[num:num+12,:].reshape(-1,12,2), y: Train_Y[num2:num2+1,:]}) 
    costs[i] = sess.run(cost, feed_dict = {x: Train_X[num:num+12,:].reshape(-1,12,2), y: Train_Y[num2:num2+1,:]})
    if i == 0 or i % 1000 == 0 or i == epoches-1:
        print(i,"epoch(es)", " Done!")
        
#------------------------generate data-----------------------------

for row in range(0, 180):
    for column in range(0, 180):
        if matrix[row, column] == 0:
            test_dis = dis_cal(row+2030, column+2630, TM_X, TM_Y).reshape(-1,1)
            input_x = np.hstack((test_dis, Z[0].reshape(-1, 1)))
            matrix[row, column] = sess.run(prediction, feed_dict = {x: input_x.reshape(-1,12,2)})

#---------------------------plot----------------------------------
            
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(2030, 2210)
Y = np.arange(2630, 2810)

x, y = np.meshgrid(X, Y)
z = matrix

ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
plt.show()

#-------------------------cost-----------------------------

plt.plot(np.arange(len(costs)), costs)
print(costs[-1])

print("總時間:",time.clock()-t0)

