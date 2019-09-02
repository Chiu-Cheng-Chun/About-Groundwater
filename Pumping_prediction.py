# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os

#------------------------------------------------------------------------------

t0 = time.clock()
#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/kaiu.ttc')
fig = plt.figure()

path = "C:\\Users\\craig\\OneDrive\\桌面\\科技部相關檔案\\高雄抽水水位(宇文)\\新增資料夾 (4)"
os.chdir(path)

pumping_all = 'pdata_combined(hour).csv'
g_file = "gdata_record(hour).csv"
location_information = '高雄-大寮(合併).xlsx'
FROM = 24 * 24 + 3 #使得資料開始的時間相同
DAYS = 10000

#定義輸出(抽水總量)
pump_data = pd.read_csv(pumping_all)           
y = pump_data.iloc[:-1, 1].as_matrix()                               #from  2017/7/29 10:00
Y = y[FROM:FROM+DAYS].reshape((-1, 1))                               #from  2017/8/22 13:00

#定義輸入(雨量和地下水位高程)
weather_data = pd.read_excel(location_information).iloc[12:,:]       #from 2017-08-22 13:00
雨量 = weather_data['降水量'].iloc[:10000].as_matrix()                #提出雨量作為輸入，並取10000筆當資料
雨量[np.isnan(雨量)] = 0
g_data = pd.read_csv(g_file).iloc[:10000]                            #from 2017-08-22 13:00
g_data_ai51 = g_data['ai51'].as_matrix()
g_data_ai52 = g_data['ai52'].as_matrix()
"""
X = np.stack((雨量, g_data_ai51, g_data_ai52), axis=1)
"""
X = np.stack((g_data_ai51, g_data_ai52), axis=1)


#偏差調整
X_bias = X[2:,:]                                                     #from 2017-08-22 15:00
Y_bias = Y[:-2,:]

#定義測試集、訓練集
x_train = X_bias[:7000, :]
y_train = Y_bias[:7000, :]
x_test = X_bias[7000:, :]
y_test = Y_bias[7000:, :]

#定義測試集、訓練集時間
train_axis = np.arange(len(x_train))
test_axis = np.arange(len(x_test))

#定義模式
DEBUG_MODE = 0


'''
輸入應該是
x_train, x_test
y_train, y_test
維度是[樣本個數,特徵數]
'''

#-------------------------------------------------------------------
if DEBUG_MODE == 0:
    input_size = x_train.shape[1]                
    output_size = y_test.shape[1]
    W = []
    b = []

    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])

    n_neurons = [input_size, 100, 100, output_size]

    for i in range(0, len(n_neurons)-1):
        W.append(tf.Variable(tf.truncated_normal([n_neurons[i], n_neurons[i+1]])))
        b.append(tf.Variable(tf.zeros([1.0, n_neurons[i+1]])))

    def model():
        res = x
        for i in range(0, len(n_neurons) - 2):
                res = tf.nn.relu(tf.matmul(res, W[i]) + b[i])        #用relu作為激勵函數
        res =  tf.matmul(res, W[-1]) + b[-1]
        return res


    def Cost(y_label, prediction):
        square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))   #選擇差平方代價函數
        return square_error


    out = model()
    cost = Cost(y, out)
    learning_rate = 0.0001
    epoches = 150000
    batch = 3000

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)              #選擇 Adam
    ACK = 0
    ACK_Threshold = 5000
    # -----------------------------------------------------------------------------------------

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for step in range(epoches+1):
        num = step % (len(x_train)-batch)
        sess.run(train_step, feed_dict={x: x_train[num:num+batch, : input_size], y: y_train[num:num+batch]})
        
        #計算ACK
        test_output = sess.run(out, feed_dict={x: x_test})
        test_sum = 0
        for i in range(len(test_output)):
            test_sum = test_sum + abs(y_test[i] - test_output[i])
        test_error = test_sum / len(test_output)
        if test_error < ACK_Threshold:
            ACK_Threshold = test_error
        elif test_error > ACK_Threshold:
            ACK+=1
        
        #每5000次印一筆資料
        if step % 5000 == 0:
            print("ACK_Threshold:", ACK_Threshold)
            print("epoches:", step)
            print("訓練集總cost:",sess.run(cost, feed_dict={x: x_train, y: y_train}))
            print("測試集總cost:",sess.run(cost, feed_dict={x: x_test, y: y_test}))
            
            #畫train set
            train_output = sess.run(out, feed_dict={x: x_train[:]})
            plt.plot(train_axis[:], y_train[:], "orange", label="pumping data")  
            plt.plot(train_axis[:], train_output, "blue", label="prediction results") 
            plt.legend(loc='upper right')
            plt.xlabel('Time(hour)')
            plt.ylabel('Pumping(m^3)')
            plt.show()
            train_output = sess.run(out, feed_dict={x: x_train})
            train_sum = 0
            for i in range(len(train_output)):
                train_sum = train_sum + abs(y_train[i] - train_output[i])
            print("Train誤差:", train_sum / len(train_output))
            print("訓練集結果↑")
            

            #畫test set
            test_output = sess.run(out, feed_dict={x: x_test[:]})
            plt.plot(test_axis[:], y_test[:], "orange", label="pumping data")  
            plt.plot(test_axis[:], test_output, "blue", label="prediction results") 
            plt.legend(loc='upper right')
            plt.xlabel('Time(hour)')
            plt.ylabel('Pumping(m^3)')
            plt.show()
            test_output = sess.run(out, feed_dict={x: x_test})
            test_sum = 0
            for i in range(len(test_output)):
                test_sum = test_sum + abs(y_test[i] - test_output[i])
            print("Test誤差:", test_sum / len(test_output))
            print("測試集結果↑")
            

    print("總共費時:",time.clock()-t0,"秒")
    
    test_output = sess.run(out, feed_dict={x: x_test})
    test_sum = 0
    for i in range(len(test_output)):
        test_sum = test_sum + abs(y_test[i] - test_output[i])
    print("誤差:", test_sum / len(test_output))

""" 
    #畫觀測井&抽水總量比較圖(偏移前)
    fig = plt.figure()
    axis = np.arange(10000)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(axis[90:140], g_data_ai51[90:140], "green", label="Well1")  
    ax1.plot(axis[90:140], g_data_ai52[90:140], "red", label="Well2")  
    ax1.set_ylabel('Groundwater Level(m)',fontsize=16)
    ax1.legend(bbox_to_anchor=(0.23, 1))
    ax2.plot(axis[90:140], Y[90:140], "blue", label="pumping data")  
    ax2.set_ylabel('Pumping(m^3)',fontsize=16)
    ax2.legend(bbox_to_anchor=(1, 1))
    ax1.set_xlabel('Time(hour)',fontsize=16)
    ax1.set_ylim(28, 33)
    ax2.set_ylim(125, 2900)
    plt.tight_layout()
    plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\抽水資料vs井.jpg",dpi=1000)
    plt.show()
    
    
    #畫濕季疊層圖           
    濕季範圍 = [7500,7600] #2018-07-01 01:00 ~ 2018-07-05 04:00
    fig,axes=plt.subplots(nrows=3,ncols=1,sharex=True,sharey=False,figsize=(10,8))
    
    axes[0].set_title("Wet Season",fontsize=16)
    axes[0].set_ylabel('Groundwater Level(m)',fontsize=16)
    axes[0].plot(np.arange(100), g_data_ai51[7500:7600],"red", label = 'Well1')
    axes[0].plot(np.arange(100), g_data_ai52[7500:7600],"blue", label = 'Well2')
    axes[0].legend(loc=4)
    
    axes[1].set_ylabel('Pumping(m^3)',fontsize=16)
    axes[1].plot(np.arange(100), Y[7500:7600],"orange")
    
    axes[2].set_ylabel('Rainfall((mm)',fontsize=16)
    axes[2].plot(np.arange(100),weather_data.iloc[7500:7600,10],"green")
    axes[2].set_ylim(0, 100)
    axes[2].set_xlim(0, 100)
    axes[2].set_xticklabels(['2018-07-01 01:00','2018-07-01 21:00', '2018-07-02 17:00','2018-07-03 13:00','2018-07-04 9:00','2018-07-05 04:00'], rotation='15',fontsize=12)
    axes[2].set_xlabel("Date",fontsize=16)
    plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\濕季圖.jpg", dpi = 500)
    plt.show()

    #畫乾季疊層圖           
    乾季範圍 = [4900,5000] #2018-03-14 17:00 ~ 2018-03-18 21:00
    fig,axes=plt.subplots(nrows=3,ncols=1,sharex=True,sharey=False,figsize=(10,8))
    
    axes[0].set_title("Dry Season",fontsize=16)
    axes[0].set_ylabel('Groundwater Level(m)',fontsize=16)
    axes[0].plot(np.arange(100), g_data_ai51[乾季範圍[0]:乾季範圍[1]],"red", label = 'Well1')
    axes[0].plot(np.arange(100), g_data_ai52[乾季範圍[0]:乾季範圍[1]],"blue", label = 'Well2')
    axes[0].legend(loc=4)
    
    axes[1].set_ylabel('Pumping(m^3)',fontsize=16)
    axes[1].plot(np.arange(100), Y[乾季範圍[0]:乾季範圍[1]],"orange")
    
    axes[2].set_ylabel('Rainfall((mm)',fontsize=16)
    axes[2].plot(np.arange(100),weather_data.iloc[乾季範圍[0]:乾季範圍[1],10],"green")
    axes[2].set_ylim(0, 100)
    axes[2].set_xlim(0, 100)
    axes[2].set_xticklabels(['2018-03-14 17:00','2018-03-15 13:00', '2018-03-16 9:00','2018-03-17 5:00','2018-03-18 1:00','2018-03-18 21:00'], rotation='15',fontsize=12)
    axes[2].set_xlabel("Date",fontsize=16)
    plt.savefig(r"C:\Users\craig\OneDrive\桌面\科技部相關檔案\乾季圖.jpg", dpi = 500)
    plt.show()
"""
    
    
    
    
    
    
    
