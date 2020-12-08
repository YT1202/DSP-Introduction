# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 04:05:32 2020

@author: YiTing
"""

import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

# 計時開始
tStart = time.time()

# 載入mnist_data & 子集合(number 2)
data_all = pd.read_csv('mnist_train.csv')
label_2 = data_all['label'] == 2
data_2 = data_all[label_2]


test_all = pd.read_csv('mnist_test.csv')
label_test_2 = test_all['label'] == 2
test_2 = test_all[label_2]


# 將label去除
data_all = data_all.drop('label', axis = 1)
data_2 = data_2.drop('label', axis = 1)

test_all = test_all.drop('label', axis = 1)
test_2 = test_2.drop('label', axis = 1)

# 將base轉為unit-length vectors
data =  np.array(data_all)
data_2 =  np.array(data_2)
B_all = preprocessing.normalize(data, norm = 'l2')
B_2 = preprocessing.normalize(data_2, norm = 'l2')

test_all =  np.array(test_all)
test_2 =  np.array(test_2)


# 定義OMP算法
def omp(sparsity,x,B):    
    residual = x  
    (M,N) = B.shape
    index = np.zeros(N,dtype = int)
    for i in range(N): 
        index[i] = -1
    signal = np.zeros((N, 1))
    for j in range(sparsity):  
        product = np.fabs(np.dot(B.T, residual))
        sl = np.argmax(product)         
        index[sl] = 1 
        inv = np.linalg.pinv(B[:,index >= 0])          
        a = np.dot(inv,x)   
        residual = x - np.dot(B[:,index >= 0], a)
    signal[index >= 0] = a
    return  signal


# 建立DCT
DCT = np.zeros((784,784))
v = range(784)
for k in range(0,784):  
    DCT_1D = np.cos(np.dot(v, k*math.pi/784))
    if k > 0:
        DCT_1D = DCT_1D - np.mean(DCT_1D)
    DCT[:,k] = DCT_1D / np.linalg.norm(DCT_1D)

print('1.a')
# HW_OMP_(1.a)===================================================================
#
sparsity = 5
B = np.dot(B_all, DCT) 
for i in range(0,31):
    x = np.dot(B_all, test_all[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_all[i].reshape(784,1)) / np.linalg.norm(test_all[i].reshape(784,1))
    print('loss in sparsity=5 : ', loss)

plt.imshow(test_all[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]') 
plt.show()
   
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=5 : number [30]')  
plt.show()
#
sparsity = 10
B = np.dot(B_all, DCT) 
for i in range(0,31):
    x = np.dot(B_all, test_all[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_all[i].reshape(784,1)) / np.linalg.norm(test_all[i].reshape(784,1))
    print('loss in sparsity=10 : ', loss)

plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=10 : number [30]')  
plt.show()
#   
sparsity = 150
B = np.dot(B_all, DCT) 
for i in range(0,31):
    x = np.dot(B_all, test_all[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_all[i].reshape(784,1)) / np.linalg.norm(test_all[i].reshape(784,1))
    locals()['loss150_' + str(i) + '_all'] = loss
    print('loss in sparsity=150 : ', loss)

plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=150 : number [30]')  
plt.show()

print('1.b')
# HW_OMP_(1.b)===================================================================
#
B_10000 = B_all[0:10000, :]
sparsity = 5
B = np.dot(B_10000, DCT) 
for i in range(0,31):
    x = np.dot(B_10000, test_all[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_all[i].reshape(784,1)) / np.linalg.norm(test_all[i].reshape(784,1))
    print('loss in sparsity=5 : ', loss)

plt.imshow(test_all[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]') 
plt.show()
     
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=5 : number [30]')  
plt.show()
#
sparsity = 10
B = np.dot(B_10000, DCT) 
for i in range(0,31):
    x = np.dot(B_10000, test_all[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_all[i].reshape(784,1)) / np.linalg.norm(test_all[i].reshape(784,1))
    print('loss in sparsity=10 : ', loss)

plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=10 : number [30]')  
plt.show()
#   
sparsity = 150
B = np.dot(B_10000, DCT) 
for i in range(0,31):
    x = np.dot(B_10000, test_all[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_all[i].reshape(784,1)) / np.linalg.norm(test_all[i].reshape(784,1))
    locals()['loss150_' + str(i) + '_all'] = loss
    print('loss in sparsity=150 : ', loss)

plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=150 : number [30]')  
plt.show()

print('2')
# HW_OMP_(2)===================================================================
#
sparsity = 10
B = np.dot(B_2, DCT) 
for i in range(0,31):
    x = np.dot(B_2, test_2[i].reshape(784,1))
    signal = omp(sparsity, x, B) 
    img_result = np.dot(DCT, signal) 
    loss = np.linalg.norm(img_result - test_2[i].reshape(784,1)) / np.linalg.norm(test_2[i].reshape(784,1))
    print('loss in sparsity=10 : ', loss)
plt.subplot(1,2,1)
plt.imshow(test_2[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]')    
plt.subplot(1,2,2)
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=10 : number [30]')  
plt.show()

print('3')
# HW_OMP_(3)===================================================================
# PCA
# 資料標準化
data_all_cen = preprocessing.normalize(data, norm = 'l2')
data_2_cen = preprocessing.normalize(data_2, norm = 'l2')


# 求協方差、特徵值、特徵向量
cov_matrix_all_cen = np.cov(data_all_cen.T)
Eig_val_all_cen, Eig_vec_all_cen = np.linalg.eig(cov_matrix_all_cen)

cov_matrix_2_cen = np.cov(data_2_cen.T)
Eig_val_2_cen, Eig_vec_2_cen = np.linalg.eig(cov_matrix_2_cen)


# (特徵值-特徵向量) & 排序
Eig_pairs_all_cen = [ (np.abs(Eig_val_all_cen[i]),Eig_vec_all_cen[:,i]) for i in range(len(Eig_val_all_cen))]

Eig_pairs_2_cen = [ (np.abs(Eig_val_2_cen[i]),Eig_vec_2_cen[:,i]) for i in range(len(Eig_val_2_cen))]

[Eig_pairs_all_cen, Eig_pairs_2_cen].sort(key = lambda x: x[0], reverse= True)

# 計算特徵值的比例_CenteredPCA
ind_var_all_cen = [(i/sum(Eig_val_all_cen))*100 for i in sorted(Eig_val_all_cen, reverse=True)] 
cum_var_all_cen = np.cumsum(ind_var_all_cen)

ind_var_2_cen = [(i/sum(Eig_val_2_cen))*100 for i in sorted(Eig_val_2_cen, reverse=True)] 
cum_var_2_cen = np.cumsum(ind_var_2_cen)


# 找各比例對應的components_CenteredPCA
cum_95_all_cen = np.where((cum_var_all_cen >= 94.5) & (cum_var_all_cen <= 95.5))
cum_50_all_cen = np.where((cum_var_all_cen >= 49.5) & (cum_var_all_cen <= 50.5))
cum_25_all_cen = np.where((cum_var_all_cen >= 24.5) & (cum_var_all_cen <= 25.5))
print(cum_95_all_cen)    # 第330個最接近95%
print(cum_50_all_cen)    # 第38個最接近50%   
print(cum_25_all_cen)    # 第7個最接近25%

cum_95_2_cen = np.where((cum_var_2_cen >= 94.5) & (cum_var_2_cen <= 95.5))
cum_50_2_cen = np.where((cum_var_2_cen >= 49.5) & (cum_var_2_cen <= 50.5))
cum_25_2_cen = np.where((cum_var_2_cen >= 24) & (cum_var_2_cen <= 26))
print(cum_95_2_cen)    # 第236個最接近95%
print(cum_50_2_cen)    # 第26個最接近50%   
print(cum_25_2_cen)    # 第5個最接近25%


# PCA _image[30] for all data_CenteredPCA
pca_all_cen_95 = PCA(n_components = 330).fit(data_all_cen)
Eig_vec_all_cen_95 = pca_all_cen_95.components_.reshape(330,28,28)
img2_all_cen_95 = (test_all[30].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_95.transpose((0,2,1)))
loss = np.linalg.norm(img2_all_cen_95[329].reshape(784,1) - test_all[30].reshape(784,1)) / np.linalg.norm(test_all[30].reshape(784,1))
print('loss_95_all : ', loss)

pca_all_cen_50 = PCA(n_components = 38).fit(data_all_cen)
Eig_vec_all_cen_50 = pca_all_cen_50.components_.reshape(38,28,28)
img2_all_cen_50 = (test_all[30].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_50.transpose((0,2,1)))
loss = np.linalg.norm(img2_all_cen_50[37].reshape(784,1) - test_all[30].reshape(784,1)) / np.linalg.norm(test_all[30].reshape(784,1))
print('loss_50_all : ', loss)

pca_all_cen_25 = PCA(n_components = 7).fit(data_all_cen)
Eig_vec_all_cen_25 = pca_all_cen_25.components_.reshape(7,28,28)
img2_all_cen_25 = (test_all[30].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_25.transpose((0,2,1)))
loss = np.linalg.norm(img2_all_cen_25[6].reshape(784,1) - test_all[30].reshape(784,1)) / np.linalg.norm(test_all[30].reshape(784,1))
print('loss_25_all : ', loss)

plt.figure(figsize=(11,7.5))
plt.subplot(1,4,1)
plt.imshow(test_all[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]')    
plt.subplot(1,4,2)
plt.imshow(img2_all_cen_25[6].reshape(28,28), cmap = 'gray')
plt.title('PCA 25% number [30]')  
plt.subplot(1,4,3)
plt.imshow(img2_all_cen_50[37].reshape(28,28), cmap = 'gray')
plt.title('PCA 50% number [30]') 
plt.subplot(1,4,4)
plt.imshow(img2_all_cen_95[329].reshape(28,28), cmap = 'gray')
plt.title('PCA 95% number [30]') 
plt.show()

# PCA _image[30] for label = 2_CenteredPCA
pca_2_cen_95 = PCA(n_components = 236).fit(data_2_cen)
Eig_vec_2_cen_95 = pca_2_cen_95.components_.reshape(236,28,28)
img2_2_cen_95 = (test_2[30].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_95.transpose((0,2,1)))
loss = np.linalg.norm(img2_2_cen_95[235].reshape(784,1) - test_2[30].reshape(784,1)) / np.linalg.norm(test_2[30].reshape(784,1))
print('loss_95_2 : ', loss)

pca_2_cen_50 = PCA(n_components = 26).fit(data_2_cen)
Eig_vec_2_cen_50 = pca_2_cen_50.components_.reshape(26,28,28)
img2_2_cen_50 = (test_2[30].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_50.transpose((0,2,1)))
loss = np.linalg.norm(img2_2_cen_50[25].reshape(784,1) - test_2[30].reshape(784,1)) / np.linalg.norm(test_2[30].reshape(784,1))
print('loss_50_2 : ', loss)

pca_2_cen_25 = PCA(n_components = 5).fit(data_2_cen)
Eig_vec_2_cen_25 = pca_2_cen_25.components_.reshape(5,28,28)
img2_2_cen_25 = (test_2[30].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_25.transpose((0,2,1)))
loss = np.linalg.norm(img2_2_cen_25[4].reshape(784,1) - test_2[30].reshape(784,1)) / np.linalg.norm(test_2[30].reshape(784,1))
print('loss_25_2 : ', loss)

plt.figure(figsize=(11,7.5))
plt.subplot(1,4,1)
plt.imshow(test_2[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]')    
plt.subplot(1,4,2)
plt.imshow(img2_2_cen_25[4].reshape(28,28), cmap = 'gray')
plt.title('PCA 25% number [30]')  
plt.subplot(1,4,3)
plt.imshow(img2_2_cen_50[25].reshape(28,28), cmap = 'gray')
plt.title('PCA 50% number [30]') 
plt.subplot(1,4,4)
plt.imshow(img2_2_cen_95[235].reshape(28,28), cmap = 'gray')
plt.title('PCA 95% number [30]') 
plt.show()


# OMP_all training data
# 25%
sparsity = 7
B = np.dot(B_all, DCT) 
x = np.dot(B_all, test_all[30].reshape(784,1))
signal = omp(sparsity, x, B) 
img_result = np.dot(DCT, signal) 
loss = np.linalg.norm(img_result - test_all[30].reshape(784,1)) / np.linalg.norm(test_all[30].reshape(784,1))
print('loss in sparsity=7 : ', loss)

plt.imshow(test_all[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]')   
plt.show()
   
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=7 : number [30]')  
plt.show()

# 50%
sparsity = 38
B = np.dot(B_all, DCT) 
x = np.dot(B_all, test_all[30].reshape(784,1))
signal = omp(sparsity, x, B) 
img_result = np.dot(DCT, signal) 
loss = np.linalg.norm(img_result - test_all[30].reshape(784,1)) / np.linalg.norm(test_all[30].reshape(784,1))
print('loss in sparsity=38 : ', loss)
  
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=38 : number [30]')  
plt.show()

# 95%
sparsity = 330
B = np.dot(B_all, DCT) 
x = np.dot(B_all, test_all[30].reshape(784,1))
signal = omp(sparsity, x, B) 
img_result = np.dot(DCT, signal) 
loss = np.linalg.norm(img_result - test_all[30].reshape(784,1)) / np.linalg.norm(test_all[30].reshape(784,1))
print('loss in sparsity=330 : ', loss)
  
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=330 : number [30]')  
plt.show()

# OMP_label = 2
# 25%
sparsity = 5
B = np.dot(B_2, DCT) 
x = np.dot(B_2, test_2[30].reshape(784,1))
signal = omp(sparsity, x, B) 
img_result = np.dot(DCT, signal) 
loss = np.linalg.norm(img_result - test_2[30].reshape(784,1)) / np.linalg.norm(test_2[30].reshape(784,1))
print('loss in sparsity=5 : ', loss)

plt.imshow(test_2[30].reshape(28,28), cmap = 'gray')
plt.title('Original number [30]')    
plt.show()
  
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=5 :number [30]')  
plt.show()

# 50%
sparsity = 26
B = np.dot(B_2, DCT) 
x = np.dot(B_2, test_2[30].reshape(784,1))
signal = omp(sparsity, x, B) 
img_result = np.dot(DCT, signal) 
loss = np.linalg.norm(img_result - test_2[30].reshape(784,1)) / np.linalg.norm(test_2[30].reshape(784,1))
print('loss in sparsity=26 : ', loss)
   
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=26 :number [30]')  
plt.show()

# 95%
sparsity = 236
B = np.dot(B_2, DCT) 
x = np.dot(B_2, test_2[30].reshape(784,1))
signal = omp(sparsity, x, B) 
img_result = np.dot(DCT, signal) 
loss = np.linalg.norm(img_result - test_2[30].reshape(784,1)) / np.linalg.norm(test_2[30].reshape(784,1))
print('loss in sparsity=236 : ', loss)
   
plt.imshow(img_result.reshape(28,28), cmap = 'gray')
plt.title('Sparsity=236 :number [30]')  
plt.show()


# 計時結束
tEnd = time.time()
print('time : ', tEnd - tStart, 's')

