# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:23:29 2020

@author: Zhoudongdong
"""

import numpy as np
import pandas as pd 


data = pd.read_csv('.../confusion_matrix_mean.csv',header=None)
data = np.asarray(data)

#calculate the kappa value 
p0 = (data[0,0]+data[1,1]+data[2,2]+data[3,3]+data[4,4])/data[5,5]
p1 = (data[0,5]*data[5,0]+data[1,5]*data[5,1]+data[2,5]*data[5,2]+data[3,5]*data[5,3]+data[4,5]*data[5,4])/(data[5,5]*data[5,5])
kappa_value = (p0-p1)/(1-p1)
kappa_value = round(kappa_value,5)


# calculate the Precision, Recall, F1-score

cm = data
cm = np.delete(cm,5,axis=0)
cm = np.delete(cm,5,axis=1)
class_num = cm.shape[0]


col_sum = np.sum(cm, axis=1)  
raw_sum = np.sum(cm, axis=0)  

TP = []  
 
for i in range(class_num):
    TP.append(cm[i, i])
 
# 计算f1-score
TP = np.array(TP)
FN = col_sum - TP
FP = raw_sum - TP

f1_m = []
for i in range(class_num):
    # f1-score
    f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
    f1_m.append(f1)
    
f1_m = np.array(f1_m)
precision = []
recall = []
f1_score = []
# precision 

for i in range(class_num):
    temp = float((TP[i]/raw_sum[i]))
    temp = round(temp,5)
    precision.append(temp)
    
# recall
for i in range(class_num):
    temp = float((TP[i] / col_sum[i]))
    temp = round(temp,5)
    recall.append(temp)
 
# f1-score
for i in range(class_num):
    temp = float((f1_m[i]))
    temp = round(temp,5)
    f1_score.append(temp)
precision = np.array(precision)
recall = np.array(recall)
f1_score = np.array(f1_score)

MF1 = np.mean(f1_score)
MF1 = round(MF1,5)


print('The mean confusion matrix:\n',cm)
print('The kappa value of confusion matrix:',kappa_value)
print('The Accuracy value of confusion matrix:',p0)
print('The precision value of confusion matrix:',precision)
print('The recall value of confusion matrix:',recall)
print('The f1_score value of confusion matrix:',f1_score)
print('The MF1 value of confusion matrix:',MF1)


