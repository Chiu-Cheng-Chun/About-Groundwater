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
import os


t0 = time.clock()

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
    matrix = np.zeros([1000,1000])
    for a in range(num_data):
        row = int(TM_X[a])-2030
        column = int(TM_Y[a])-26300
        matrix[row, column] = Z[i][a]
    return matrix

Data = pd.read_excel("地下水位(日)_台中地區_測試.xlsx", header=None)
Data = np.array(Data)
TM_X = Data[0,4:-2] / 100
TM_Y = Data[1,4:-2] / 100
num_data = len(TM_X)
Z = np.arange(len(TM_X))
for i in range(31):
    Z =  np.vstack((Z, Data[1464+i,4:-2]))
Z = Z[1:,:]

matrix = make_matrix(Z, num_data, TM_X, TM_Y)
train_dis = pre_dis_cal(TM_X, TM_Y)


Train_X = np.zeros((1,24))
for i in range(12):
    Train_X = np.vstack((  Train_X, np.hstack((train_dis[i], Z[0]))  ))
Train_X = Train_X[1:,:]
Train_Y = Z[0].reshape(-1,1)

#------------------------------------------------------ANN--------------------------------------------------------

input_size = 24
output_size = 1

W = []
b = []

x = tf.placeholder(tf.float32,[None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

n_neurons = [input_size, 30, 30, output_size]

for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))

def model():
    res = x
    for i in range(0, len(n_neurons) - 2):
        res = tf.nn.sigmoid(tf.matmul(res,W[i]) + b[i])
    res =  tf.matmul(res,W[-1]) + b[-1]
    return res

def Cost(y_label, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))
    return square_error


out = model()
cost = Cost(y, out)
learning_rate = 0.001
epoches = 25000
batch = 1

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#---------------------------train---------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
costs = np.arange(epoches+1)

for i in range(epoches+1):
    num = i % (len(Train_X) - batch)
    sess.run(train_step, feed_dict = {x: Train_X[num:num+batch], y: Train_Y[num:num+batch]})
    costs[i] = sess.run(cost, feed_dict = {x: Train_X[num:num+batch], y: Train_Y[num:num+batch]})
    if i % 1000 == 0:
        print("epoches", i)
        
#------------------------generate data-----------------------------

for row in range(0, 1000):
    for column in range(0, 1000):
        if matrix[row, column] == 0:
            test_dis = dis_cal(row+2030, column+26300, TM_X, TM_Y)
            input_x = np.hstack((test_dis, Z[0].reshape(1, 12))).reshape(1, 24)
            matrix[row, column] = sess.run(out, feed_dict = {x: input_x})


#---------------------------plot----------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(2030, 3030)
Y = np.arange(26300, 27300)

x, y = np.meshgrid(X, Y)
z = matrix

ax.plot_surface(x, y, z.T, rstride=1, cstride=1, cmap='rainbow')
plt.show()

#-------------------------cost-----------------------------


#plt.plot(np.arange(len(costs)), costs)
print(costs[-1])
print("總時間:",time.clock()-t0)