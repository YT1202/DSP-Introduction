# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 01:52:30 2020

@author: YiTing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 載入mnist_data & 子集合(number 2 and number 7)
data_all = pd.read_csv('mnist_train.csv')

label_2 = data_all['label'] == 2
label_7 = data_all['label'] == 7

data_2 = data_all[label_2]
data_7 = data_all[label_7]



# 將label去除
data_all = data_all.drop('label', axis = 1)
data_2 = data_2.drop('label', axis = 1)
data_7 = data_7.drop('label', axis = 1)



# HW_PCA_(1)===================================================================

# 三組資料取平均
mean_all = np.mean(data_all.values, axis = 0)
mean_2 = np.mean(data_2.values, axis = 0)
mean_7 = np.mean(data_7.values, axis = 0)



# 作圖_3張_(28,28)
plt.figure(figsize = (8,8))
plt.subplot(1,3,1)
plt.imshow(mean_all.reshape(28,28), cmap = 'gray')
plt.title('Mean of all data')
#plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(mean_2.reshape(28,28), cmap = 'gray')
plt.title('Mean of number 2')
#plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(mean_7.reshape(28,28), cmap = 'gray')
plt.title('Mean of number 7')
#plt.axis('off')
plt.show()

# =============================================================================



# 資料標準化
data_all_cen = StandardScaler().fit_transform(data_all.values)
data_all_non = StandardScaler(with_mean = False).fit_transform(data_all.values)

data_2_cen = StandardScaler().fit_transform(data_2.values)
data_2_non = StandardScaler(with_mean = False).fit_transform(data_2.values)

data_7_cen = StandardScaler().fit_transform(data_7.values)
data_7_non = StandardScaler(with_mean = False).fit_transform(data_7.values)



# HW_PCA_(2)===================================================================

# PCA & EigenVector for all data
pca_all_cen = PCA(n_components = 1).fit(data_all_cen)
Eig_vec_all_cen_1 = pca_all_cen.components_.reshape(1,28,28)

pca_all_cen = PCA(n_components = 2).fit(data_all_cen)
Eig_vec_all_cen_2 = pca_all_cen.components_.reshape(2,28,28)

pca_all_cen = PCA(n_components = 3).fit(data_all_cen)
Eig_vec_all_cen_3 = pca_all_cen.components_.reshape(3,28,28)



# PCA & EigenVector for number 2
pca_2_cen = PCA(n_components = 1).fit(data_2_cen)
Eig_vec_2_cen_1 = pca_2_cen.components_.reshape(1,28,28)

pca_2_cen = PCA(n_components = 2).fit(data_2_cen)
Eig_vec_2_cen_2 = pca_2_cen.components_.reshape(2,28,28)

pca_2_cen = PCA(n_components = 3).fit(data_2_cen)
Eig_vec_2_cen_3 = pca_2_cen.components_.reshape(3,28,28)



# PCA & EigenVector for number 3
pca_7_cen = PCA(n_components = 1).fit(data_7_cen)
Eig_vec_7_cen_1 = pca_7_cen.components_.reshape(1,28,28)

pca_7_cen = PCA(n_components = 2).fit(data_7_cen)
Eig_vec_7_cen_2 = pca_7_cen.components_.reshape(2,28,28)

pca_7_cen = PCA(n_components = 3).fit(data_7_cen)
Eig_vec_7_cen_3 = pca_7_cen.components_.reshape(3,28,28)



# 作圖_9張_(28,28)
plt.figure(figsize=(13,13))
plt.subplot(3,3,1)
plt.imshow(Eig_vec_all_cen_1[0].reshape(28,28), cmap = 'gray')
plt.title('1st Eigenvector for all data', size = 10)

plt.subplot(3,3,2)
plt.imshow(Eig_vec_all_cen_2[1].reshape(28,28), cmap = 'gray')
plt.title('2nd Eigenvector for all data', size = 10)

plt.subplot(3,3,3)
plt.imshow(Eig_vec_all_cen_3[2].reshape(28,28), cmap = 'gray')
plt.title('3rd Eigenvector for all data', size = 10)

plt.subplot(3,3,4)
plt.imshow(Eig_vec_2_cen_1[0].reshape(28,28), cmap = 'gray')
plt.title('1st Eigenvector for number 2', size = 10)

plt.subplot(3,3,5)
plt.imshow(Eig_vec_2_cen_2[1].reshape(28,28), cmap = 'gray')
plt.title('2nd Eigenvector for number 2', size = 10)

plt.subplot(3,3,6)
plt.imshow(Eig_vec_2_cen_3[2].reshape(28,28), cmap = 'gray')
plt.title('3rd Eigenvector for number 2', size = 10)

plt.subplot(3,3,7)
plt.imshow(Eig_vec_7_cen_1[0].reshape(28,28), cmap = 'gray')
plt.title('1st Eigenvector for number 7', size = 10)

plt.subplot(3,3,8)
plt.imshow(Eig_vec_7_cen_2[1].reshape(28,28), cmap = 'gray')
plt.title('2nd Eigenvector for number 7', size = 10)

plt.subplot(3,3,9)
plt.imshow(Eig_vec_7_cen_3[2].reshape(28,28), cmap = 'gray')
plt.title('3rd Eigenvector for number 7', size = 10)
plt.show()
 
# =============================================================================



# HW_PCA_(3)===================================================================

# 求協方差、特徵值、特徵向量
cov_matrix_all_cen = np.cov(data_all_cen.T)
cov_matrix_all_non = np.cov(data_all_non.T)
Eig_val_all_cen, Eig_vec_all_cen = np.linalg.eig(cov_matrix_all_cen)
Eig_val_all_non, Eig_vec_all_non = np.linalg.eig(cov_matrix_all_non)

cov_matrix_2_cen = np.cov(data_2_cen.T)
cov_matrix_2_non = np.cov(data_2_non.T)
Eig_val_2_cen, Eig_vec_2_cen = np.linalg.eig(cov_matrix_2_cen)
Eig_val_2_non, Eig_vec_2_non = np.linalg.eig(cov_matrix_2_non)

cov_matrix_7_cen = np.cov(data_7_cen.T)
cov_matrix_7_non = np.cov(data_7_non.T)
Eig_val_7_cen, Eig_vec_7_cen = np.linalg.eig(cov_matrix_7_cen)
Eig_val_7_non, Eig_vec_7_non = np.linalg.eig(cov_matrix_7_non)



# (特徵值-特徵向量) & 排序
Eig_pairs_all_cen = [ (np.abs(Eig_val_all_cen[i]),Eig_vec_all_cen[:,i]) for i in range(len(Eig_val_all_cen))]
Eig_pairs_all_non = [ (np.abs(Eig_val_all_non[i]),Eig_vec_all_non[:,i]) for i in range(len(Eig_val_all_non))]

Eig_pairs_2_cen = [ (np.abs(Eig_val_2_cen[i]),Eig_vec_2_cen[:,i]) for i in range(len(Eig_val_2_cen))]
Eig_pairs_2_non = [ (np.abs(Eig_val_2_non[i]),Eig_vec_2_non[:,i]) for i in range(len(Eig_val_2_non))]

Eig_pairs_7_cen = [ (np.abs(Eig_val_7_cen[i]),Eig_vec_7_cen[:,i]) for i in range(len(Eig_val_7_cen))]
Eig_pairs_7_non = [ (np.abs(Eig_val_7_non[i]),Eig_vec_7_non[:,i]) for i in range(len(Eig_val_7_non))]

[Eig_pairs_all_cen, Eig_pairs_all_non, 
  Eig_pairs_2_cen, Eig_pairs_2_non, 
  Eig_pairs_7_cen, Eig_pairs_7_non].sort(key = lambda x: x[0], reverse= True)



# 計算特徵值的比例_CenteredPCA
ind_var_all_cen = [(i/sum(Eig_val_all_cen))*100 for i in sorted(Eig_val_all_cen, reverse=True)] 
cum_var_all_cen = np.cumsum(ind_var_all_cen)

ind_var_2_cen = [(i/sum(Eig_val_2_cen))*100 for i in sorted(Eig_val_2_cen, reverse=True)] 
cum_var_2_cen = np.cumsum(ind_var_2_cen)

ind_var_7_cen = [(i/sum(Eig_val_7_cen))*100 for i in sorted(Eig_val_7_cen, reverse=True)] 
cum_var_7_cen = np.cumsum(ind_var_7_cen)



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

cum_95_7_cen = np.where((cum_var_7_cen >= 94.5) & (cum_var_7_cen <= 95.5))
cum_50_7_cen = np.where((cum_var_7_cen >= 49.5) & (cum_var_7_cen <= 50.5))
cum_25_7_cen = np.where((cum_var_7_cen >= 24) & (cum_var_7_cen <= 26))
print(cum_95_7_cen)    # 第223個最接近95%
print(cum_50_7_cen)    # 第22個最接近50%   
print(cum_25_7_cen)    # 第4個最接近25%



# PCA _image[2] for all data_CenteredPCA
pca_all_cen_95 = PCA(n_components = 330).fit(data_all_cen)
Eig_vec_all_cen_95 = pca_all_cen_95.components_.reshape(330,28,28)
img2_all_cen_95 = (data_all_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_95.transpose((0,2,1)))

pca_all_cen_50 = PCA(n_components = 38).fit(data_all_cen)
Eig_vec_all_cen_50 = pca_all_cen_50.components_.reshape(38,28,28)
img2_all_cen_50 = (data_all_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_50.transpose((0,2,1)))

pca_all_cen_25 = PCA(n_components = 7).fit(data_all_cen)
Eig_vec_all_cen_25 = pca_all_cen_25.components_.reshape(7,28,28)
img2_all_cen_25 = (data_all_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_25.transpose((0,2,1)))



# PCA _image[27] for all data_CenteredPCA
pca_all_cen_95 = PCA(n_components = 330).fit(data_all_cen)
Eig_vec_all_cen_95 = pca_all_cen_95.components_.reshape(330,28,28)
img27_all_cen_95 = (data_all_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_95.transpose((0,2,1)))

pca_all_cen_50 = PCA(n_components = 38).fit(data_all_cen)
Eig_vec_all_cen_50 = pca_all_cen_50.components_.reshape(38,28,28)
img27_all_cen_50 = (data_all_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_50.transpose((0,2,1)))

pca_all_cen_25 = PCA(n_components = 7).fit(data_all_cen)
Eig_vec_all_cen_25 = pca_all_cen_25.components_.reshape(7,28,28)
img27_all_cen_25 = (data_all_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_all_cen_25.transpose((0,2,1)))



# 作圖_95,50,25_6張_(28,28) for image[2] & image[27]  all data_CenteredPCA
plt.figure(figsize=(11,7.5))
plt.subplot(2,3,1)
plt.imshow(img2_all_cen_25[6].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[2] for all data', size = 10)

plt.subplot(2,3,2)
plt.imshow(img2_all_cen_50[37].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[2] for all data', size = 10)

plt.subplot(2,3,3)
plt.imshow(img2_all_cen_95[329].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[2] for all data', size = 10)

plt.subplot(2,3,4)
plt.imshow(img27_all_cen_25[6].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[27] for all data', size = 10)

plt.subplot(2,3,5)
plt.imshow(img27_all_cen_50[37].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[27] for all data', size = 10)

plt.subplot(2,3,6)
plt.imshow(img27_all_cen_95[329].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[27] for all data', size = 10)
plt.show()



# PCA _image[2] for number 2_CenteredPCA
pca_2_cen_95 = PCA(n_components = 236).fit(data_2_cen)
Eig_vec_2_cen_95 = pca_2_cen_95.components_.reshape(236,28,28)
img2_2_cen_95 = (data_2_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_95.transpose((0,2,1)))

pca_2_cen_50 = PCA(n_components = 26).fit(data_2_cen)
Eig_vec_2_cen_50 = pca_2_cen_50.components_.reshape(26,28,28)
img2_2_cen_50 = (data_2_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_50.transpose((0,2,1)))

pca_2_cen_25 = PCA(n_components = 5).fit(data_2_cen)
Eig_vec_2_cen_25 = pca_2_cen_25.components_.reshape(5,28,28)
img2_2_cen_25 = (data_2_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_25.transpose((0,2,1)))



# PCA _image[27] for number 2_CenteredPCA
pca_2_cen_95 = PCA(n_components = 236).fit(data_2_cen)
Eig_vec_2_cen_95 = pca_2_cen_95.components_.reshape(236,28,28)
img27_2_cen_95 = (data_2_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_95.transpose((0,2,1)))

pca_2_cen_50 = PCA(n_components = 26).fit(data_2_cen)
Eig_vec_2_cen_50 = pca_2_cen_50.components_.reshape(26,28,28)
img27_2_cen_50 = (data_2_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_50.transpose((0,2,1)))

pca_2_cen_25 = PCA(n_components = 5).fit(data_2_cen)
Eig_vec_2_cen_25 = pca_2_cen_25.components_.reshape(5,28,28)
img27_2_cen_25 = (data_2_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_2_cen_25.transpose((0,2,1)))



# 作圖_95,50,25_6張_(28,28) for image[2] & image[27]  number 2_CenteredPCA
plt.figure(figsize=(11,7.5))
plt.subplot(2,3,1)
plt.imshow(img2_2_cen_25[4].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[2] for number 2', size = 10)

plt.subplot(2,3,2)
plt.imshow(img2_2_cen_50[25].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[2] for number 2', size = 10)

plt.subplot(2,3,3)
plt.imshow(img2_2_cen_95[235].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[2] for number 2', size = 10)

plt.subplot(2,3,4)
plt.imshow(img27_2_cen_25[4].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[27] for number 2', size = 10)

plt.subplot(2,3,5)
plt.imshow(img27_2_cen_50[25].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[27] for number 2', size = 10)

plt.subplot(2,3,6)
plt.imshow(img27_2_cen_95[235].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[27] for number 2', size = 10)
plt.show()



# PCA _image[2] for number 7_CenteredPCA
pca_7_cen_95 = PCA(n_components = 223).fit(data_7_cen)
Eig_vec_7_cen_95 = pca_7_cen_95.components_.reshape(223,28,28)
img2_7_cen_95 = (data_7_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_7_cen_95.transpose((0,2,1)))

pca_7_cen_50 = PCA(n_components = 22).fit(data_7_cen)
Eig_vec_7_cen_50 = pca_7_cen_50.components_.reshape(22,28,28)
img2_7_cen_50 = (data_7_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_7_cen_50.transpose((0,2,1)))

pca_7_cen_25 = PCA(n_components = 4).fit(data_7_cen)
Eig_vec_7_cen_25 = pca_7_cen_25.components_.reshape(4,28,28)
img2_7_cen_25 = (data_7_cen[2].reshape(28,28)[None,:,:])*(Eig_vec_7_cen_25.transpose((0,2,1)))



# PCA _image[27] for number 7_CenteredPCA
pca_7_cen_95 = PCA(n_components = 223).fit(data_7_cen)
Eig_vec_7_cen_95 = pca_7_cen_95.components_.reshape(223,28,28)
img27_7_cen_95 = (data_7_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_7_cen_95.transpose((0,2,1)))

pca_7_cen_50 = PCA(n_components = 22).fit(data_7_cen)
Eig_vec_7_cen_50 = pca_7_cen_50.components_.reshape(22,28,28)
img27_7_cen_50 = (data_7_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_7_cen_50.transpose((0,2,1)))

pca_7_cen_25 = PCA(n_components = 4).fit(data_7_cen)
Eig_vec_7_cen_25 = pca_7_cen_25.components_.reshape(4,28,28)
img27_7_cen_25 = (data_7_cen[27].reshape(28,28)[None,:,:])*(Eig_vec_7_cen_25.transpose((0,2,1)))



# 作圖_95,50,25_6張_(28,28) for image[2] & image[27]  number 7_CenteredPCA
plt.figure(figsize=(11,7.5))
plt.subplot(2,3,1)
plt.imshow(img2_7_cen_25[3].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[2] for number 7', size = 10)

plt.subplot(2,3,2)
plt.imshow(img2_7_cen_50[21].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[2] for number 7', size = 10)

plt.subplot(2,3,3)
plt.imshow(img2_7_cen_95[222].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[2] for number 7', size = 10)

plt.subplot(2,3,4)
plt.imshow(img27_7_cen_25[3].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[27] for number 7', size = 10)

plt.subplot(2,3,5)
plt.imshow(img27_7_cen_50[21].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[27] for number 7', size = 10)

plt.subplot(2,3,6)
plt.imshow(img27_7_cen_95[222].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[27] for number 7', size = 10)
plt.show()
# =============================================================================



# HW_PCA_(4)===================================================================

# 計算特徵值的比例_Non-CenteredPCA
ind_var_all_non = [(i/sum(Eig_val_all_non))*100 for i in sorted(Eig_val_all_non, reverse=True)] 
cum_var_all_non = np.cumsum(ind_var_all_non)

ind_var_2_non = [(i/sum(Eig_val_2_non))*100 for i in sorted(Eig_val_2_non, reverse=True)] 
cum_var_2_non = np.cumsum(ind_var_2_non)



# 找各比例對應的components_Non-CenteredPCA
cum_95_all_non = np.where((cum_var_all_non >= 94.5) & (cum_var_all_non <= 95.5))
cum_50_all_non = np.where((cum_var_all_non >= 49.5) & (cum_var_all_non <= 50.5))
cum_25_all_non = np.where((cum_var_all_non >= 24.5) & (cum_var_all_non <= 25.5))
print(cum_95_all_non)    # 第329個最接近95%
print(cum_50_all_non)    # 第38個最接近50%   
print(cum_25_all_non)    # 第7個最接近25%

cum_95_2_non = np.where((cum_var_2_non >= 94.5) & (cum_var_2_non <= 95.5))
cum_50_2_non = np.where((cum_var_2_non >= 49.5) & (cum_var_2_non <= 50.5))
cum_25_2_non = np.where((cum_var_2_non >= 24) & (cum_var_2_non <= 26))
print(cum_95_2_non)    # 第236個最接近95%
print(cum_50_2_non)    # 第26個最接近50%   
print(cum_25_2_non)    # 第5個最接近25%



# PCA _image[2] for all data_Non-CenteredPCA
pca_all_non_95 = PCA(n_components = 329).fit(data_all_non)
Eig_vec_all_non_95 = pca_all_non_95.components_.reshape(329,28,28)
img2_all_non_95 = (data_all_non[2].reshape(28,28)[None,:,:])*(Eig_vec_all_non_95.transpose((0,2,1)))

pca_all_non_50 = PCA(n_components = 38).fit(data_all_non)
Eig_vec_all_non_50 = pca_all_non_50.components_.reshape(38,28,28)
img2_all_non_50 = (data_all_non[2].reshape(28,28)[None,:,:])*(Eig_vec_all_non_50.transpose((0,2,1)))

pca_all_non_25 = PCA(n_components = 7).fit(data_all_non)
Eig_vec_all_non_25 = pca_all_non_25.components_.reshape(7,28,28)
img2_all_non_25 = (data_all_non[2].reshape(28,28)[None,:,:])*(Eig_vec_all_non_25.transpose((0,2,1)))



# PCA _image[2] for number 2_Non-CenteredPCA
pca_2_non_95 = PCA(n_components = 236).fit(data_2_non)
Eig_vec_2_non_95 = pca_2_non_95.components_.reshape(236,28,28)
img2_2_non_95 = (data_2_non[2].reshape(28,28)[None,:,:])*(Eig_vec_2_non_95.transpose((0,2,1)))

pca_2_non_50 = PCA(n_components = 26).fit(data_2_non)
Eig_vec_2_non_50 = pca_2_non_50.components_.reshape(26,28,28)
img2_2_non_50 = (data_2_non[2].reshape(28,28)[None,:,:])*(Eig_vec_2_non_50.transpose((0,2,1)))

pca_2_non_25 = PCA(n_components = 5).fit(data_2_non)
Eig_vec_2_non_25 = pca_2_non_25.components_.reshape(5,28,28)
img2_2_non_25 = (data_2_non[2].reshape(28,28)[None,:,:])*(Eig_vec_2_non_25.transpose((0,2,1)))



# 作圖_95,50,25_6張_(28,28) for image[2]  all data & number 2_Non-CenteredPCA
plt.figure(figsize=(11,7.5))
plt.subplot(2,3,1)
plt.imshow(img2_all_non_25[6].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[2] for all data', size = 10)

plt.subplot(2,3,2)
plt.imshow(img2_all_non_50[37].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[2] for all data', size = 10)

plt.subplot(2,3,3)
plt.imshow(img2_all_non_95[328].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[2] for all data', size = 10)

plt.subplot(2,3,4)
plt.imshow(img2_2_non_25[4].reshape(28,28), cmap = 'gray')
plt.title('25% energy : Image[2] for number 2', size = 10)

plt.subplot(2,3,5)
plt.imshow(img2_2_non_50[25].reshape(28,28), cmap = 'gray')
plt.title('50% energy : Image[2] for number 2', size = 10)

plt.subplot(2,3,6)
plt.imshow(img2_2_non_95[235].reshape(28,28), cmap = 'gray')
plt.title('95% energy : Image[2] for number 2', size = 10)
plt.show()
# =============================================================================
