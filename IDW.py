# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:07:11 2019

@author: craig
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:52:41 2018

@author: YU-TING
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

def Find_losing_point(array_input):
    total=array_input.shape[0]
    point_index=[]
    
    for i in range(total):
        if(array_input[i]==0):
            point_index.append(i)
    
    return point_index

def Computing_Distance(point_index,array_input):
    distance=[]
    total=array_input.shape[0]

    for i in point_index:
        point_distance=[]
        for j in range(total):
            
            if(j in point_index):
                point_distance.append(0)    
                
            else:
                point_distance.append(abs(i-j))
        distance.append(point_distance)
        
    return distance

def Computing_Value(point_index,array_input,distance,exp=-1):
    distance=np.array(distance,dtype='float64')
    mask=distance==0
    mx=np.ma.masked_array(distance,mask=mask)
    inv_distance=1/mx               #check
    inv_distance=inv_distance.data-inv_distance.mask
    total=np.sum(inv_distance,axis=1,keepdims=True)
    W=inv_distance/total
    Value=W.T*array_input
    Value=np.sum(Value,axis=0,keepdims=True)
    return Value

def Value_Interpolation(Value,point_index,array_input):
    point_index=np.array(point_index)
    Value=Value.T
    number=0
    for i in point_index:
        array_input[i,0]=Value[number,0]
        number+=1
    return array_input

def Idw_Interpolation(A_array_input,exp=-1):
    point_index=Find_losing_point(A_array_input)
    distance=Computing_Distance(point_index,A_array_input)
    Value=Computing_Value(point_index,A_array_input,distance)
    data=Value_Interpolation(Value,point_index,A_array_input)
    return data

#-----------------------------------------------------------------
    
os.chdir("C:\\Users\\craig\\OneDrive\\桌面")
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

for i in range(100):
    height[500+i] = 0

height = np.array(height)
height = height.reshape(1,-1)
data = Idw_Interpolation(height)