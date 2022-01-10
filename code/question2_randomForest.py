from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from scipy.linalg.interpolative import seed
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import pickle
import random
import os

random.seed(2021)

root_path = '/home/bigdata9/wxs/game/math_model/code'
label_coloumn = ['pIC50']
train_path = root_path+'/data/train_Molecular_ER.xlsx'

data = pd.read_excel(train_path,sheet_name='Sheet1')
data = data.sample(frac=1,random_state=2021).reset_index(drop=True)
columns_list = data.columns.values.tolist()
# X = data[columns_list[1:len(columns_list)-2]]


if not os.path.exists('./Descriptor_activaty.pkl'):
    Molecular_Descriptor_test = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/Molecular_Descriptor.xlsx", sheet_name="test")
    ER_alpha_activity_test = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/ERα_activity.xlsx", sheet_name="test")
    test = Molecular_Descriptor_test.merge(ER_alpha_activity_test,on=['SMILES'],how='left')
    pickle.dump(test,open('./Descriptor_activaty.pkl','wb'))

test_data = pickle.load(open('./Descriptor_activaty.pkl','rb'))

feature_name = open('./entropy_feat_threshold_0.txt','r').read().split('\n')[:-1] 
##421
t = 0.05  ## 421
feat_421 = pd.read_csv('./feature_entropy_pierxun_spearman_divide_score_select_'+str(t)+'.csv').columns.values.tolist()
feature_name = feat_421
# tg_feature_name = pd.read_csv('./tg_final_out.csv') 
# feature_name = tg_feature_name.columns.values.tolist()
# # ## lgb 20 importance features
# feature_name = ['MDEC-13','maxHBd','ETA_Eta_B_RC','ATSc5','SPC-6','ETA_Shape_Y','MLFER_E','MLFER_A','SC-5','ATSc4','XLogP','minHBint10','mindO','maxHaaCH','ALogP','ATSp5','VPC-5','VCH-7','minHsOH','maxwHBa']
# ## random forest 20 importance features
# feature_name = ['MDEC-23','minsssN','maxssO','maxHsOH','C1SP2','BCUTc-1l','minHsOH','LipoaffinityIndex','SHsOH','nHBAcc','minsOH','ATSc3','VC-5','MDEO-12','CrippenLogP','TopoPSA','BCUTc-1h','nC','XLogP','MLFER_A']

# ##2021.10.18 rf_lgb_vote 
# feature_name = ['BCUTc-1l', 'ATSc5', 'XLogP', 'ATSc4', 'LipoaffinityIndex','SPC-6', 'VC-5', 'ATSc3', 'ALogP', 'MLFER_A', 'SdssC', 'minssCH2','MDEC-23', 'hmin', 'BCUTp-1l', 'MDEC-33', 'minHsOH', 'BCUTc-1h','minHBint10', 'VCH-7']
# ##           rf top 20
# feature_name = ['MDEC-23','LipoaffinityIndex','minsssN','minHsOH','maxHsOH','C1SP2','maxssO','BCUTc-1l','nHBAcc','minHBint5'	,'ATSc3'	,'VC-5','MLFER_A','mindssC','MDEC-33','TopoPSA','ndssC','ATSc2','BCUTp-1h','nHBAcc_Lipinski']
# ##           lgb top 20
# feature_name = ['BCUTc-1l','ATSc5'	,'ATSc4',	'XLogP'	,'LipoaffinityIndex'	,'SPC-6'	,'VC-5'	,'ATSc3'	,'ALogP'	,'MLFER_A'	,'SdssC'	,'minssCH2',	'hmin'	,'BCUTp-1l'	,'BCUTc-1h',	'minHBint10',	'MDEC-33'	,'VCH-7',	'minHsOH'	,'SsOH']
# feature_name = ['MDEC-23', 'LipoaffinityIndex', 'BCUTc-1l', 'ATSc5', 'XLogP',
#        'ATSc4', 'minHsOH', 'minsssN', 'ATSc3', 'VC-5', 'SPC-6', 'MLFER_A',
#        'maxHsOH', 'ALogP', 'SdssC', 'hmin', 'minssCH2', 'MDEC-33',
#        'BCUTc-1h', 'minHBint10']
X = data[feature_name]
y = data[columns_list[-1]]

data_length = len(y)
split_num = int(data_length*0.8)
trn_data = X[:split_num]
# trn_data = X[idx[:split_num]]
trn_y = y[:split_num]
val_data = X[split_num:]
val_y = y[split_num:]

# X_train, Y_train, X_test, Y_test = pickle.load(open('./X_Y.pickle','rb'))
# trn_data,trn_y,val_data,val_y = X_train[feature_name],Y_train,X_test[feature_name],Y_test


# Create the model with 100 trees
model = RandomForestRegressor(n_estimators=500, bootstrap = True,random_state=44)
# Fit on training data
model.fit(trn_data, trn_y)

feat_importance = model.feature_importances_
result = pd.DataFrame([feat_importance],columns=feature_name)
result.to_csv('./feature_entropy_randomForest_score.csv',index=False)
with open('./randomForest_feature_importance.txt','w') as f:
    for item in feat_importance.tolist():
        f.write(str(item))
        f.write('\n')

val_predict = model.predict(val_data)
loss = mean_squared_error(val_predict,val_y)
print('mse',loss) ##mse 1.9351880244200954   ## 0.64

# ER_alpha_activity_test = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/ERα_activity.xlsx", sheet_name="test")

# ER_alpha_activity_test[label_coloumn] = model.predict(test_data[feature_name])

# ER_alpha_activity_test.to_csv('./randomForest_test_predict.csv',index=False)

