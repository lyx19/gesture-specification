# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:10:49 2018

@author: 123
"""

#os模块中包含很多操作文件和目录的函数 
import os 
import numpy as np
#获取目标文件夹的路径 
meragefiledir = os.getcwd()+'\\五指张'
#获取当前文件夹中的文件名称列表 
filenames=os.listdir(meragefiledir) 
#打开当前目录下的result.txt文件，如果没有则创建
file=open('result_t4.txt','w') 
#向文件中写入字符 
  
#先遍历文件名 
for filename in filenames: 
  filepath=meragefiledir+'\\'
  filepath=filepath+filename
  #遍历单个文件，读取行数 
  for line in open(filepath):
    file.writelines(str(line)) 
  file.write('\n') 
#关闭文件 
file.close()


