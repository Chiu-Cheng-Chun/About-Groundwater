# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:03:10 2019

@author: craig
"""

import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\craig\\OneDrive\\桌面")
height = pd.read_excel("地下水位(日)_台中地區(2013.01_2016.12).xlsx")

Y = height.iloc[173446:174542, 9]
height = np.array(['地下水位高程'], dtype = np.object).reshape(1,1)
Y = np.array(Y, dtype = np.object).reshape(-1,1)
Y = np.vstack((height, Y))

#--------------------------------------------------------------------

os.chdir("C:\\Users\\craig\\OneDrive\\桌面\\台中氣象站")

def get_total():    
    total_content = []
    file = os.walk("C:\\Users\\craig\\OneDrive\\桌面\\台中氣象站")
    for path,dir_list,file_list in file:  
        for file_name in file_list:  
            content = pd.read_excel(file_name)
            total_content = np.append(total_content, content)
        return total_content.reshape(-1,17)
    
def count(total_content):
    final = np.zeros((1096,6))
    title = ['平均測站氣壓','平均氣溫','平均相對溼度','平均風速','降水總量','全天空日射總量']
    for i in range(1096):
        final[i, 0] = np.mean(total_content[i*24:(i+1)*24, 1])
        final[i, 1] = np.mean(total_content[i*24:(i+1)*24, 3])
        final[i, 2] = np.mean(total_content[i*24:(i+1)*24, 5])
        final[i, 3] = np.mean(total_content[i*24:(i+1)*24, 6])
        final[i, 4] = np.sum(total_content[i*24:(i+1)*24, 10])
        final[i, 5] = np.sum(total_content[i*24:(i+1)*24, 13])
    final = np.array(final, dtype = np.object)
    title = np.array(title, dtype = np.object)
    final = np.vstack((title, final))
    return final

def delete_nan(total_content):
    for i in range(len(total_content)):
        if np.isnan(total_content[i,10]):
            total_content[i,10] = 0
    return total_content

def save(final, Y):
    os.chdir("C:\\Users\\craig\\OneDrive\\桌面")
    final = np.hstack((final, Y))
    file_dataframe = pd.DataFrame(final)
    file_dataframe.to_excel('台中氣象站(日資料).xlsx', index = False, header = None)
    print("Done!")
    
total_content = get_total()
total_content = delete_nan(total_content)
final = count(total_content)
save(final,Y)