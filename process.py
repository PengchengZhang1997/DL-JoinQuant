# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:33:24 2020

@author: zpc
"""

import numpy as np
import pandas as pd

timesteps = 8
dataset = pd.read_csv('train_etf.csv')
y_price = dataset['price'].values
train_y = []
train_y.append([0,0])
train_y=np.array(train_y)
del dataset['price']
train_x=dataset.values
x_train = np.array([])
y_train = np.array([])
index=0

while(index+timesteps<=1249*11):
    if(y_price[index+timesteps-1]==1):
        train_y[0][1]=1
        train_y[0][0]=0
    else:
        train_y[0][1]=0
        train_y[0][0]=1
    x_train = np.append(x_train, train_x[index: index + timesteps, :])
    y_train = np.append(y_train, train_y[0])
    if ((index+timesteps)%1249 == 0):
        print(index)
        index=index+timesteps
    else:
        index=index+1

pdx=pd.DataFrame(x_train)
pdy=pd.DataFrame(y_train)
pdx.to_csv('train_x11.csv',index=None)
pdy.to_csv('train_y11.csv',index=None)