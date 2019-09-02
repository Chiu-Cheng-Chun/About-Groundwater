# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:38:11 2018

@author: craig
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tensorflow as tf
import time
import os
import copy

os.chdir('C:\\Users\\craig\\Downloads')

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
        distance[0, column] = np.sqrt((Y - TM_X[column])**2 + (X - TM_Y[column])**2)  
    return distance

def make_matrix(Z, num_data, TM_X, TM_Y):
    matrix = np.zeros([1000,1000])
    high = np.zeros((12,1))
    i = 0
    for a in range(num_data):
        column = int(TM_X[a])-2030
        row = int(TM_Y[a])-26300
        matrix[row, column] = Z[i][a]
        high[i] = Map2[row, column]
        i+=1
    return matrix, high

def rotate(list1, num):
    b = [[0 for i in range(num)] for j in range(num)]
    for i in range(num):
        for j in range(num):
            b[num-1-j][num-1-i] = list1[i][num-1-j]
    return b

Data = pd.read_excel("地下水位(日)_台中地區_測試.xlsx", header=None)
Data = np.array(Data)
TM_X = Data[0,4:-2] / 100
TM_Y = Data[1,4:-2] / 100
num_data = len(TM_X)
Z = np.arange(len(TM_X))
for i in range(31):
    Z =  np.vstack((Z, Data[1464+i,4:-2]))
Z = Z[1:,:]

matrix, high = make_matrix(Z, num_data, TM_X, TM_Y)
train_dis = pre_dis_cal(TM_X, TM_Y)

Train_X = np.zeros((1,24))
for i in range(12):
    Train_X = np.vstack((  Train_X, np.hstack((train_dis[i], Z[0]))  ))
Train_X = np.hstack((  Train_X[1:,:], high  ))
Train_Y = Z[0].reshape(-1,1)

#------------------------------------------------------ANN--------------------------------------------------------

input_size = 25
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
learning_rate = 0.0003
epoches = 30000
batch = 12
zero_times = 0
Done_place = 0

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#---------------------------train---------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
costs = []

for i in range(epoches+1):
    num = i % (len(Train_X) - batch + 1)
    sess.run(train_step, feed_dict = {x: Train_X[num:num+batch], y: Train_Y[num:num+batch]})
    c = sess.run(cost, feed_dict = {x: Train_X[num:num+batch], y: Train_Y[num:num+batch]})
    costs = np.append(costs, c)
    if c <= 0.0001:
        zero_times = zero_times + 1
        if zero_times == 10:
            Done_place = i
            break
    if i % 1000 == 0:
        print("epoch", i)
        
#------------------------generate data-----------------------------
print("Stating generating data...")
for row in range(0, 1000):
    for column in range(0, 180):
        if matrix[row, column] == 0:
            test_dis = dis_cal(row+26300, column+2030, TM_X, TM_Y)
            input_x = np.hstack((test_dis, Z[0].reshape(1, 12))).reshape(1, 24)
            input_x = np.hstack((input_x, Map2[row, column].reshape(1,1)))
            matrix[row, column] = sess.run(out, feed_dict = {x: input_x})
print("Done!")

#-------------------------time&save-----------------------------

np.save('2D插植-2.npy', a)
print("總時間:",time.clock()-t0)

print("Start ploting...")
X = np.arange(2030, 3030)
Y = np.arange(26300, 27300)
x, y = np.meshgrid(X, Y)
z = np.array(matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(y, x, z, rstride=1, cstride=1, cmap='Oranges', linewidth=10.0)
ax.set_ylabel('TM_X(100m)')
ax.set_xlabel('TM_Y(100m)')
ax.set_zlabel('Groundwater Level(m)')
ax.set_title("Observation Data")
plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\水位高(插值).jpg", dpi = 1000)
plt.show()

print("Done!")


plot = 0
if plot:
    #---------------------------plot 水位3D----------------------------------
    print("Start ploting...")
    X = np.arange(2030, 3030)
    Y = np.arange(26300, 27300)
    x, y = np.meshgrid(X, Y)
    z = np.array(matrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(y, x, z, rstride=1, cstride=1, cmap='Oranges', linewidth=10.0)
    ax.set_ylabel('TM_X(100m)')
    ax.set_xlabel('TM_Y(100m)')
    ax.set_zlabel('Groundwater Level(m)')
    ax.set_title("Observation Data")
    plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\水位高.jpg", dpi = 1000)
    plt.show()
    
    print("Done!")
    
    
    #----------------plot 水位2D--------------------------------------------
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    水位高 = np.sum(z, 1)
    for i in range(len(水位高)):
        if 水位高[i] == 0:
            水位高[i] = np.nan
    ax1.scatter(np.arange(26200,27200), 水位高)
    ax1.set_xlabel('TM_Y(100m)',fontsize=12)
    ax1.set_ylabel('Groundwater Level(m)',fontsize=12)
    ax1.set_title("Observation Data",fontsize=12)
    plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\水位高2D.jpg", dpi = 1000)
    plt.show()
    
    
    #----------------2019/07/20 for地勢圖--------------------
    print("Start ploting...")
    X = np.arange(2030, 3030)
    Y = np.arange(26300, 27300)
    x, y = np.meshgrid(X, Y)
    
    z = copy.deepcopy(Map2)
    
    for i in range(100,1000):
        z[:,i] = z[:,i] * 0
    z = np.multiply(z > 0, z) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(y, x, z, rstride=1, cstride=1, cmap='OrRd')
    ax.set_ylabel('TM_X(100m)')
    ax.set_xlabel('TM_Y(100m)')
    ax.set_zlabel('Height(m)')
    ax.set_title("Observation Data")
    plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\地勢高.jpg", dpi = 1000)
    plt.show()
    
    print("Done!")
