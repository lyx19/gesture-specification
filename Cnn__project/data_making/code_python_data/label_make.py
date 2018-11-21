# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:09:30 2018

@author: 123
"""

import numpy as np
# 导入数据集   ## f.txt : 410x2400    f_label.txt : 410x10

a = np.zeros((1,220),  dtype=np.int)
b = np.ones((1,200),  dtype=np.int)
c = np.ones((1,100), dtype=np.int)*2
d = np.ones((1,140), dtype=np.int)*3
e = np.ones((1,120), dtype=np.int)*4

f = np.zeros((1,40), dtype=np.int)
g = np.ones((1,20), dtype=np.int)
h = np.ones((1,20), dtype=np.int)*2
i = np.ones((1,20), dtype=np.int)*3
j = np.ones((1,20), dtype=np.int)*4
values = np.hstack((a,b,c,d,e,f,g,h,i,j))

labels = np.eye(5)[values]
labels = labels.reshape((900, -1))  #1000 * 10
labels = labels.astype(np.int8)
label_f0 = labels[0:220, :]
label_f1 = labels[220:420, :]
label_f2 = labels[420:520, :]
label_f3 = labels[520:660, :]
label_f4 = labels[660:780, :]
label_f5 = labels[780:820, :]
label_f6 = labels[820:840, :]
label_f7 = labels[840:860, :]
label_f8 = labels[860:880, :]
label_f9 = labels[880:, :]

file_0 = open('label_t0.txt', 'wb')
np.savetxt(file_0, label_f0, fmt="%g",delimiter=" ")
file_0.close()

