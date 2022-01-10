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
import os

random.seed(2021)
max_features = 50

root_path = '/home/bigdata9/wxs/game/math_model/code'
label_coloumn = ['pIC50']
train_path = root_path+'/data/train_Molecular_ER.xlsx'

if not os.path.exists('./Descriptor_activaty.pkl'):
    Molecular_Descriptor_test = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/Molecular_Descriptor.xlsx", sheet_name="test")
    ER_alpha_activity_test = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/ERα_activity.xlsx", sheet_name="test")
    test = Molecular_Descriptor_test.merge(ER_alpha_activity_test,on=['SMILES'],how='left')
    pickle.dump(test,open('./Descriptor_activaty.pkl','wb'))

test_data = pickle.load(open('./Descriptor_activaty.pkl','rb'))


data = pd.read_excel(train_path,sheet_name='Sheet1')
data = data.sample(frac=1,random_state=2021).reset_index(drop=True)
# data = pickle.load(open('./activity_and_descriptor_train.pkl','rb'))
columns_list = data.columns.values.tolist()



feature_name = open('./entropy_feat_threshold_0.txt','r').read().split('\n')[:-1]
# tg_feature_name = pd.read_csv('./tg_final_out.csv') 
# feature_name = tg_feature_name.columns.values.tolist()
#421
t = 0.05  ## 421
feat_421 = pd.read_csv('./feature_entropy_pierxun_spearman_divide_score_select_'+str(t)+'.csv').columns.values.tolist()
feature_name = feat_421
## nn feature  0.76
# feature_name =['PetitjeanNumber', 'SHBint10', 'minsOH', 'maxsOH', 'nT9Ring','minssO', 'minaaS', 'SaasN', 'minaaN', 'maxaasC', 'ETA_EtaP_B_RC','maxssO', 'nHCsatu', 'minaaCH', 'maxHBint5', 'maxHBint3','MDEC-12', 'minddssS', 'nBase', 'maxHBint2']
## lgb 20 importance features
# feature_name = ['MDEC-13','maxHBd','ETA_Eta_B_RC','ATSc5','SPC-6','ETA_Shape_Y','MLFER_E','MLFER_A','SC-5','ATSc4','XLogP','minHBint10','mindO','maxHaaCH','ALogP','ATSp5','VPC-5','VCH-7','minHsOH','maxwHBa']
## random forest 20 importance features  0.657
# feature_name = ['MDEC-23','minsssN','maxssO','maxHsOH','C1SP2','BCUTc-1l','minHsOH','LipoaffinityIndex','SHsOH','nHBAcc','minsOH','ATSc3','VC-5','MDEO-12','CrippenLogP','TopoPSA','BCUTc-1h','nC','XLogP','MLFER_A']
## 四个基础投票 20 
# feature_name = ['SP-5','maxsssN','SwHBa','AMR','C2SP2','n6Ring','nT6Ring','minHsOH','MDEC-22','C1SP2','minsssN','nC','MLogP','BCUTp-1h','CrippenLogP','minsOH','maxsOH','hmin','LipoaffinityIndex','MDEC-23']

##2021.10.18 rf_lgb_vote 
# feature_name = ['BCUTc-1l', 'ATSc5', 'XLogP', 'ATSc4', 'LipoaffinityIndex','SPC-6', 'VC-5', 'ATSc3', 'ALogP', 'MLFER_A', 'SdssC', 'minssCH2','MDEC-23', 'hmin', 'BCUTp-1l', 'MDEC-33', 'minHsOH', 'BCUTc-1h','minHBint10', 'VCH-7']
# ##           rf top 20
# feature_name = ['MDEC-23','LipoaffinityIndex','minsssN','minHsOH','maxHsOH','C1SP2','maxssO','BCUTc-1l','nHBAcc','minHBint5'	,'ATSc3'	,'VC-5','MLFER_A','mindssC','MDEC-33','TopoPSA','ndssC','ATSc2','BCUTp-1h','nHBAcc_Lipinski']
# ##           lgb top 20
# feature_name = ['BCUTc-1l','ATSc5'	,'ATSc4',	'XLogP'	,'LipoaffinityIndex'	,'SPC-6'	,'VC-5'	,'ATSc3'	,'ALogP'	,'MLFER_A'	,'SdssC'	,'minssCH2',	'hmin'	,'BCUTp-1l'	,'BCUTc-1h',	'minHBint10',	'MDEC-33'	,'VCH-7',	'minHsOH'	,'SsOH']
# feature_name = ['MDEC-23', 'LipoaffinityIndex', 'BCUTc-1l', 'ATSc5', 'XLogP',
#        'ATSc4', 'minHsOH', 'minsssN', 'ATSc3', 'VC-5', 'SPC-6', 'MLFER_A',
#        'maxHsOH', 'ALogP', 'SdssC', 'hmin', 'minssCH2', 'MDEC-33',
#        'BCUTc-1h', 'minHBint10']

# X = data[columns_list[1:len(columns_list)-2]]
X = data[feature_name]
y = data[columns_list[-1]]

data_length = len(y)
idx = [i for i in range(data_length)]
split_num = int(data_length*0.8)
# X = X.sample(frac=1).reset_index(drop=True)
# y = y.sample(frac=1).reset_index(drop=True)
trn_data = X[:split_num]
# trn_data = X[idx[:split_num]]
trn_y = y[:split_num]
val_data = X[split_num:]
val_y = y[split_num:]

# X_train, Y_train, X_test, Y_test = pickle.load(open('./X_Y.pickle','rb'))
# trn_data,trn_y,val_data,val_y = X_train[feature_name],Y_train,X_test[feature_name],Y_test


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

# with open(f'model_{loss}.pkl', 'wb') as handle:
#             pickle.dump(lgb_model, handle)


best_iter = lgb_model.best_iteration_

features = X.columns.values.tolist()
importances = lgb_model.feature_importances_
result = np.argsort(np.abs(importances))[-max_features:]

# with open('./lgb_eval_loss.txt','w') as f_loss:
#     for item in eval_result:
#         f_loss.write(str(item))
#         f_loss.write('\n')

# result_lgb = pd.DataFrame([importances],columns=feature_name)
# result_lgb.to_csv('./feature_entropy_lgb_score.csv',index=False)

# with open('./feature_importance_lgb.txt','w') as f:
#     for item in importances:
#         print(item)
#         f.write(str(item))
#         f.write('\n')
# f.close()

# with open('./lgb_feature.txt','a') as f:
#     f.write(str(np.array(importances)[result]))
#     f.write('\n')
#     f.write(str(np.array(features)[result]))
#     f.write('\n')
#     f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     f.write('\n')

print('runtime: {}\n'.format(time.time() - t))




# #################### 全量训练 #####################
# #带有验证集验证效果


t = time.time()

clf = LGBMRegressor(
    learning_rate=0.05,
    n_estimators=best_iter,
    num_leaves=255,
    subsample=1,
    colsample_bytree=1,
    random_state=2021,
    metric='None'
)

clf.fit(
    X, y,
    eval_set=[(X, y)],
    verbose=100
)


ER_alpha_activity_test = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/ERα_activity.xlsx", sheet_name="test")

ER_alpha_activity_test[label_coloumn] = clf.predict(test_data[feature_name])

ER_alpha_activity_test.to_csv('./question2_lgb_test_predict.csv',index=False)
print('runtime: {}\n'.format(time.time() - t))


