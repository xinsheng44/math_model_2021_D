from lightgbm import LGBMRegressor
from scipy.linalg.interpolative import seed
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import pickle
import random
import matplotlib.pyplot as plt
import lightgbm

random.seed(2021)
max_features = 20

root_path = '/home/bigdata9/wxs/game/math_model/code'
label_coloumn = ['pIC50']
train_path = root_path+'/data/train_Molecular_ER.xlsx'

data = pd.read_excel(train_path,sheet_name='Sheet1')
columns_list = data.columns.values.tolist()

weights = [1,1,1,1]
feature_score = pickle.load(open('./result_socre'+'_'+str(weights[0])+'_'+str(weights[1])+'_'+str(weights[2])+'_'+str(weights[3])+'.pkl','rb'))

columns_score = np.array(columns_list)[np.argsort(feature_score)[-max_features:]].tolist()
print(columns_score)
# feat = ['minHBint9', 'C1SP2', 'minaaN', 'gmax', 'nT5Ring', 'ALogp2', 'SHBa', 'C2SP1', 'minssssNp', 'SdsCH', 'maxssssNp', 'VP-7', 'gmin', 'nAtomLAC', 'MDEC-22', 'BCUTp-1l', 'VP-3', 'VP-5', 'VP-6']
   

# X = data[columns_list[1:len(columns_list)-2]]
X = data[columns_score]
# X = data[feat]
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

##################### 线下验证 #####################
uauc_list = []
r_list = []


t = time.time()

lgb_model = LGBMRegressor(
    learning_rate=0.05,
    n_estimators=1000,
    num_leaves=255,
    subsample=1,
    colsample_bytree=1,
    random_state=2021,
    metric='None'
)

lgb_model.fit(
    trn_data, trn_y,
    eval_set=[(val_data, val_y)],
    eval_metric='mse',
    early_stopping_rounds=100,
    verbose=50,
)

# score = lgb_model.score(trn_data,trn_y)
# print(score)
eval_result = lgb_model.evals_result_['valid_0']['l2']
val_predict = lgb_model.predict(val_data)
loss = mean_squared_error(val_predict,val_y)
print('mse',loss) 

with open(f'model_feat_test_{loss}.pkl', 'wb') as handle:
            pickle.dump(lgb_model, handle)


best_iter = lgb_model.best_iteration_

features = X.columns.values.tolist()
importances = lgb_model.feature_importances_
result = np.argsort(np.abs(importances))[-max_features:]








"""

['minHBint9' 'C1SP2' 'minaaN' 'gmax' 'nT5Ring' 'ALogp2' 'SHBa' 'C2SP1'
 'minssssNp' 'SdsCH' 'maxssssNp' 'nB' 'VP-7' 'gmin' 'nAtomLAC' 'MDEC-22'
 'BCUTp-1l' 'VP-3' 'VP-5' 'VP-6']
 mse  1.7588 1 1 1 1

 ['minaaN' 'gmax' 'ALogp2' 'minssssNp' 'SHBa' 'fragC' 'C2SP1' 'maxssssNp'
 'minHBint9' 'nB' 'SdsCH' 'VPC-5' 'VP-7' 'gmin' 'nAtomLAC' 'MDEC-22'
 'BCUTp-1l' 'VP-3' 'VP-5' 'VP-6']
 mse 1.7836  1122

 ['gmax' 'C2SP1' 'SHBa' 'minssssNp' 'nB' 'maxssssNp' 'SdsCH' 'fragC'
 'maxHBint10' 'minHBint9' 'LipinskiFailures' 'VP-7' 'nAtomLAC' 'gmin'
 'VPC-5' 'MDEC-22' 'BCUTp-1l' 'VP-3' 'VP-5' 'VP-6']
 mse 1.7778  2121

['gmax' 'C2SP1' 'SHBa' 'minssssNp' 'nB' 'maxssssNp' 'SdsCH' 'fragC'
 'maxHBint10' 'minHBint9' 'LipinskiFailures' 'VP-7' 'nAtomLAC' 'gmin'
 'VPC-5' 'MDEC-22' 'BCUTp-1l' 'VP-3' 'VP-5' 'VP-6']
 mse 1.7778 4121

 """


