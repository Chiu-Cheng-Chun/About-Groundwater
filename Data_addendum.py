# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os
import copy

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/kaiu.ttc')
fig = plt.figure()
t0 = time.clock()


#------------------------------------------------------------------------------


os.chdir("C:\\Users\\craig\\OneDrive\\桌面\\科技部相關檔案")
data = pd.read_excel("台中氣象站(日資料).xlsx")

input_X = data.iloc[:-1,1:-1]
output_Y = data.iloc[:,-1]
height = output_Y

for num in range(0,4):
    input_X.iloc[:,num] = input_X.iloc[:,num] * 10 / max(input_X.iloc[:,num])
output_Y = np.diff(output_Y)

train_X = np.vstack((input_X.iloc[:500], input_X.iloc[600:]))
test_X = np.array(input_X.iloc[500:600])
train_Y = np.vstack((output_Y[:500].reshape(-1,1), output_Y[600:].reshape(-1,1)))
test_Y = output_Y[500:600].reshape(-1,1)

date = data.iloc[:,0]

DEBUG_MODE = 0

'''
輸入應該是
x_train, x_test
y_train, y_test
維度是[樣本個數,特徵數]
'''
#-------------------------------------------------------------------
if DEBUG_MODE == 0:
        input_size = input_X.shape[1]                
        output_size = 1
        keep_prob = 0.8
        W = []
        b = []

        x = tf.placeholder(tf.float32, [None, input_size])
        y = tf.placeholder(tf.float32, [None, output_size])

        n_neurons = [input_size, 50, 50, output_size]

        for i in range(0, len(n_neurons)-1):
                W.append(tf.Variable(tf.truncated_normal([n_neurons[i], n_neurons[i+1]])))
                b.append(tf.Variable(tf.zeros([1.0, n_neurons[i+1]])))

        def model():
                res = x
                for i in range(0, len(n_neurons) - 2):
                        res = tf.nn.sigmoid(tf.matmul(res, W[i]) + b[i])        #用relu作為激勵函數
                        #res = tf.nn.dropout(res, keep_prob)
                res =  tf.matmul(res, W[-1]) + b[-1]
                return res

        def Cost(y_label, prediction):
                square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))   #選擇差平方代價函數
                return square_error

        out = model()
        cost = Cost(y, out)
        learning_rate = 0.0001
        epoches = 10000
        batch = 1000

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)              #選擇 Adam
        # ---------------------------

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        costs = np.arange(epoches+1)

        for i in range(epoches+1):
                num = i % (len(train_X)-batch)
                sess.run(train_step, feed_dict={x: train_X[num:num+batch], y: train_Y[num:num+batch]})
                costs[i] = sess.run(cost, feed_dict = {x: train_X[num:num+batch], y: train_Y[num:num+batch]})
                
                if i % 100 == 0:
                        print("\nepoches", i)
                        print('Cost:',sess.run(cost, feed_dict={x: train_X[num:num+batch], y: train_Y[num:num+batch]}))
                        outs = sess.run(out, feed_dict={x: test_X[:]})    

                        x_axis = np.arange(len(test_Y))
                        plt.plot(x_axis, test_Y, "orange", label="real deviation")  
                        plt.plot(x_axis, outs, "green", label="prediction deviation") 
                        plt.legend(loc='upper right')
                        plt.title("insertion")
                        plt.show()
                        
                        
        
        final = np.zeros((100,1))
        last = height[500]
        x_axis = np.arange(len(height))
        for num in range(100):
            final[num] = last + outs[num]
            last = final[num]
            
        middle = copy.deepcopy(height) 
        height[500:600] = np.nan
        middle[:499] = np.nan
        middle[601:] = np.nan
        plt.plot(date, height, "orange", label="observation data")  
        plt.plot(date, middle, "red", label="observation data")
        plt.legend(loc='upper right')
        plt.xlabel('Date(day)')
        plt.ylabel('Groundwater Level(m)')   
        plt.gcf().autofmt_xdate()
        plt.show()
        
        error = 0
        for num in range(100):
            error = error + abs(final[num] - height[500+num])
        print('Error=', error)
        
             
        height[500:600] = np.nan
        middle = copy.deepcopy(height) 
        middle[:499] = np.nan
        middle[601:] = np.nan
        for num in range(100):
             middle[500+num] = Idw[500+num]
        plt.plot(date, height, "green", label="observation data")
        plt.plot(date, middle, "red", label="prediction results")
        plt.legend(loc='upper right')
        plt.xlabel('Date(day)')
        plt.ylabel('Groundwater Level(m)')
        plt.gcf().autofmt_xdate()
        plt.show()
        
        
        plt.plot(date[1:], output_Y, "green", label="中山(1)地下水位變化量")
        plt.legend(loc='upper right')
        plt.xlabel('Date(day)')
        plt.ylabel('中山(1)地下水位變化量(m)')
        plt.gcf().autofmt_xdate()
        plt.show()
        
        
        print("總共費時:",time.clock()-t0,"秒")
