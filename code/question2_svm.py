import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import random

seed = 2021
random.seed(seed)
max_features = 50

root_path = '/home/bigdata9/wxs/game/math_model/code'
label_coloumn = ['pIC50']
train_path = root_path+'./data/train_Molecular_ER.xlsx'

Molecular_Descriptor_train = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/Molecular_Descriptor.xlsx", sheet_name="training")
ER_alpha_activity_train = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/ERα_activity.xlsx", sheet_name="training")
train = Molecular_Descriptor_train.merge(ER_alpha_activity_train,on=['SMILES'],how='left')
train.to_excel(train_path,index=False)

data = pd.read_excel(train_path,sheet_name='Sheet1')
columns_list = data.columns.values.tolist()
X = data[columns_list[1:len(columns_list)-2]]
y = data[columns_list[-1]]

data_length = len(y)
idx = [i for i in range(data_length)]
split_num = int(data_length*0.2)
# X = X.sample(frac=1,random_state=2021).reset_index(drop=True)
# y = y.sample(frac=1,random_state=2021).reset_index(drop=True)
trn_data = X[:split_num]
# trn_data = X[idx[:split_num]]
trn_y = y[:split_num]
val_data = X[split_num:]
val_y = y[split_num:]


##只有线性核才能得到特征重要性分数

print ('SVR - RBF')
svr_rbf = svm.SVR(kernel='rbf', gamma=0.4, C=100)
svr_rbf.fit(trn_data, trn_y)

print ('SVR - Polynomial')
svr_poly = svm.SVR(kernel='poly', degree=3, C=100)
svr_poly.fit(trn_data, trn_y)

print ('SVR - Linear')
svr_linear = svm.SVR(kernel='linear', C=100)
svr_linear.fit(trn_data, trn_y)

print ('Fit OK.')



y_rbf = svr_rbf.predict(val_data)
y_linear = svr_linear.predict(val_data)
y_poly = svr_poly.predict(val_data)

print("高斯核函数支持向量机的平均绝对误差为:", mean_absolute_error(val_y,y_rbf))
print("高斯核函数支持向量机的均方误差为:", mean_squared_error(val_y,y_rbf))
print("多项式核函数支持向量机的平均绝对误差为:", mean_absolute_error(val_y,y_poly))
print("多项式核函数支持向量机的均方误差为:", mean_absolute_error(val_y,y_poly))

"""
高斯核函数支持向量机的平均绝对误差为: 1.4492899444498073
高斯核函数支持向量机的均方误差为: 3.032858536620456
多项式核函数支持向量机的平均绝对误差为: 6.806169861192964
多项式核函数支持向量机的均方误差为: 6.806169861192964
"""