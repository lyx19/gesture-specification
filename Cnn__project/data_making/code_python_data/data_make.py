# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:02:44 2018

@author: 123
"""

open('train_data.txt', 'w')
open('train_label.txt', 'w')
open('test_data.txt', 'w')
open('test_label.txt', 'w') 

import numpy as np
# 导入数据集   ## f.txt : 410x2400    f_label.txt : 410x10
'''
data_ff = np.loadtxt('result_f0.txt')
data_fd = np.loadtxt('result_f1.txt') 
data_fl = np.loadtxt('result_f2.txt')
data_fr = np.loadtxt('result_f3.txt')
data_fs = np.loadtxt('result_f4.txt')

data_fz = np.loadtxt('result_f5.txt')
data_tf = np.loadtxt('result_f6.txt')
data_td = np.loadtxt('result_f7.txt')
data_tl = np.loadtxt('result_f8.txt')
data_tr = np.loadtxt('result_f9.txt')

data_ff = data_ff.reshape((100, -1))
data_fd = data_fd.reshape((100, -1))
data_fl = data_fl.reshape((100, -1))
data_fr = data_fr.reshape((100, -1))
data_fs = data_fs.reshape((100, -1))

data_fz = data_ff.reshape((100, -1))
data_tf = data_fd.reshape((100, -1))
data_td = data_fl.reshape((100, -1))
data_tl = data_fr.reshape((100, -1))
data_tr = data_fs.reshape((100, -1))
'''
#python字典的应用，str格式可以写循环str(i)
'''
k = 6
data_dict = {'data_t'+str(i): np.loadtxt('result_t'+str(i)+'.txt') for i in range(k)}
data = np.vstack((data_dict["data_t"+str(i)] for i in range(k)))
data = data.reshape((400, -1))

label_dict = {'label_t'+str(i): np.loadtxt('label_t'+str(i)+'.txt') for i in range(k)}

label = np.vstack((label_dict["label_t"+str(i)] for i in range(k)))

data_label = np.hstack((data, label))

np.random.shuffle(data_label)

train_data_label = data_label[0:300]
test_data_label = data_label[300:]

train_data,a1,a2 = np.array_split(train_data_label, (3600, 3601), axis = 1)
train_label = np.hstack((a1,a2))
#train_data = train_data.reshape((2400, -1))

test_data,b1,b2 = np.array_split(test_data_label, (3600, 3601), axis = 1)
test_label = np.hstack((b1,b2))
#test_data = test_data.reshape((600, -1))
np.savetxt('train_data.txt', train_data, fmt = "%g")
np.savetxt('test_data.txt', test_data, fmt = "%g")
np.savetxt('train_label.txt', train_label, fmt = "%d")
np.savetxt('test_label.txt', test_label, fmt = "%d")
'''

k = 5
data_dict = {'data_t'+str(i): np.loadtxt('result_t'+str(i)+'.txt') for i in range(k)}
data = np.vstack((data_dict["data_t"+str(i)] for i in range(k)))
data = data.reshape((780, -1))

test_dict = {'test_t'+str(i): np.loadtxt('test_t'+str(i)+'.txt') for i in range(k)}
test = np.vstack((test_dict["test_t"+str(i)] for i in range(k)))
test = test.reshape((120, -1))

label_dict = {'label_t'+str(i): np.loadtxt('label_t'+str(i)+'.txt') for i in range(k)}
label = np.vstack((label_dict["label_t"+str(i)] for i in range(k)))

test_label_dict = {'test_label_t'+str(i): np.loadtxt('test_label_t'+str(i)+'.txt') for i in range(k)}
test_label = np.vstack((test_label_dict["test_label_t"+str(i)] for i in range(k)))

train_data_label = np.hstack((data, label))

test_data_label = np.hstack((test, test_label))

np.random.shuffle(train_data_label)
np.random.shuffle(test_data_label)

train_data,a1,a2 = np.array_split(train_data_label, (3600, 3601), axis = 1)
train_label = np.hstack((a1,a2))

test_data,b1,b2 = np.array_split(test_data_label, (3600, 3601), axis = 1)
test_label = np.hstack((b1,b2))

np.savetxt('train_data.txt', train_data, fmt = "%g")
np.savetxt('test_data.txt', test_data, fmt = "%g")
np.savetxt('train_label.txt', train_label, fmt = "%d")
np.savetxt('test_label.txt', test_label, fmt = "%d")


