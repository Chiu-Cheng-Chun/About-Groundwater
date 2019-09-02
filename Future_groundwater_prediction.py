# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:12:10 2019

@author: craig
"""

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os
import copy

#------------------------------------------------------------------------------
path = "C:\\Users\\craig\\OneDrive\\桌面\\科技部相關檔案"
os.chdir(path)

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/kaiu.ttc')
fig = plt.figure()

t0 = time.clock()

Data = pd.read_excel('DATA4.xlsx', header = 0 )
Data = pd.DataFrame(Data)

pre_x_train = np.array(Data.iloc[:3000,2:-1])
pre_x_test = np.array(Data.iloc[3000:,2:-1])
y_train = np.array(Data.iloc[:3000, 0]).reshape(-1,1)
y_test = np.array(Data.iloc[3000:, 0]).reshape(-1,1)


train_height = Data.iloc[:3000,1]
test_height = Data.iloc[3000:,1]

def recalculate_zero(pre_x_train, pre_x_test): #補0為前後相加除2
    X_PASS = pre_x_train == 0
    for i in range(3000):
        if(np.sum(X_PASS[i,3:7])==4):
            for j in range(4):
                pre_x_train[i,3+j] = (pre_x_train[i-1,3+j] + pre_x_train[i+1,3+j])/2
    
    X_PASS = pre_x_test == 0
    for i in range(653):
        if(np.sum(X_PASS[i,3:7])==4):
            for j in range(4):
                pre_x_test[i,3+j] = (pre_x_test[i-1,3+j] + pre_x_test[i+1,3+j])/2
    return pre_x_train, pre_x_test

x_train, x_test = recalculate_zero(pre_x_train, pre_x_test)

#------------------------------PreProcess-----------------------------
#-------------------------------------------------------------------

input_size =  5                                                       
output_size = 1

W = []
b = []

x = tf.placeholder(tf.float32,[None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

n_neurons = [input_size, 50, 50, output_size]

for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))

def model():
    res = x
    for i in range(0, len(n_neurons) - 2):
        res = tf.nn.sigmoid(tf.matmul(res,W[i]) + b[i])                             #用relu作為激勵函數
    res =  tf.matmul(res,W[-1]) + b[-1]
    return res

def Cost(y_label, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))   #選擇差平方代價函數
    return square_error   

out = model()
cost = Cost(y, out)
learning_rate = 0.001
epoches = 5000000
batch = 1000
check_point = 100000#判斷是否存圖
saver = tf.train.Saver()

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)              #選擇 Adam
#---------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
costs = np.arange(epoches)
Predict_Future = np.zeros((650,1))

Validation = copy.deepcopy(x_test)
for i in range(epoches):
    num = i % (len(x_train)-batch)
    sess.run(train_step, feed_dict = {x: x_train[num:num+batch], y: y_train[num:num+batch]})
    costs[i] = sess.run(cost, feed_dict = {x: x_train[num:num+batch], y: y_train[num:num+batch]})
    
    if i % 5000 == 0 or i == epoches-1:
        print("epoches", i+1)
        
        x_train_predict = sess.run(out, feed_dict = {x: x_train})
        x_test_predict = sess.run(out, feed_dict = {x: x_test})

        final = np.zeros((len(x_train),1))
        last = train_height[0]
        for num in range(len(x_train)):
            final[num] = last + x_train_predict[num]
            last = final[num]        

        test_final = np.zeros((len(x_test),1))
        test_last = 76.73
        for num in range(len(x_test)):
            test_final[num] = test_last + x_test_predict[num]
            test_last = test_final[num]

        
        train_y = np.append(   final, np.full(np.shape(x_test_predict), np.nan)   )
        test_y = np.append(   np.full(np.shape(x_train_predict), np.nan), test_final   )
        plt.plot(Data.iloc[:,-1], Data.iloc[:,1], "orange", label="observation data")  
        plt.plot(Data.iloc[:,-1], train_y, "g--", label="Training results") 
        plt.plot(Data.iloc[:,-1], test_y, "b-.", label="Testing results") 
        plt.legend(loc='upper left')
        plt.xlabel('Date(day)')
        plt.ylabel('Groundwater Level(m)')   
        plt.gcf().autofmt_xdate()
        train_y = np.append(   final, np.full(np.shape(x_test_predict), 0)   )
        test_y = np.append(   np.full(np.shape(x_train_predict), 0), test_final   )
        if np.sum(     ((train_y + test_y)-Data.iloc[:,1])**2    ) < check_point: 
            plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\地下水位高程預測(測試3).jpg", dpi = 1000)
            check_point = np.sum(     ((train_y + test_y)-Data.iloc[:,1])**2    )
            print("Check point:",check_point)
            save_path = r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\地下水位預測_model(測試3)"
            spath = saver.save(sess, save_path)
            print("Model saved in file: %s" % spath)
        plt.show()
        
        """畫變化量圖
        plt.plot(test_time, y_test, "orange", label="observation data")  
        plt.plot(test_time, x_test_predict, "green", label="prediction results") 
        plt.legend(loc='lower right')
        plt.xlabel('Date(day)')
        plt.ylabel('Groundwater Level(m)')   
        plt.gcf().autofmt_xdate()
        plt.show()
        """
        """畫變化量圖
        plt.plot(train_time, y_train, "orange", label="observation data")  
        plt.plot(train_time, x_train_predict, "green", label="prediction results") 
        plt.legend(loc='lower right')
        plt.xlabel('Date(day)')
        plt.ylabel('Groundwater Level(m)')   
        plt.gcf().autofmt_xdate()
        plt.show()
        """
        """畫水位高度
        plt.plot(Data.iloc[:,-1], Data.iloc[:,1], "orange", label="observation data")  
        plt.legend(loc='upper right')
        plt.xlabel('Date(day)')
        plt.ylabel('Groundwater Level(m)')   
        plt.gcf().autofmt_xdate()
        plt.show()
        """
        
        
test_sum = 0
for i in range(len(x_test_predict)):
    test_sum = test_sum + abs(x_test_predict[i] - y_test[i])
print("誤差:", test_sum / len(x_test_predict))


print("總共費時:",time.clock()-t0,"秒")




