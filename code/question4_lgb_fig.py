from typing import DefaultDict
from lightgbm import LGBMClassifier
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
# from lightgbm.sklearn import LGBMClassifier

random.seed(2021)

root_path = '/home/bigdata9/wxs/game/math_model'
if not os.path.exists(root_path+'/Descriptor_ADMET_train.pkl'):
    Molecular_Descriptor_train = pd.read_excel("./code/data/Molecular_Descriptor.xlsx", sheet_name="training")
    ADMET_train = pd.read_excel("./code/data/ADMET.xlsx", sheet_name="training")
    train = Molecular_Descriptor_train.merge(ADMET_train,on=['SMILES'],how='left')
    pickle.dump(train,open('./Descriptor_ADMET_train.pkl','wb'))

if not os.path.exists(root_path+'/Descriptor_ADMET_test.pkl'):
    Molecular_Descriptor_train = pd.read_excel("./code/data/Molecular_Descriptor.xlsx", sheet_name="test")
    ADMET_test = pd.read_excel("./code/data/ADMET.xlsx", sheet_name="test")
    test = Molecular_Descriptor_train.merge(ADMET_test,on=['SMILES'],how='left')
    pickle.dump(test,open('./Descriptor_ADMET_test.pkl','wb'))

data = pickle.load(open(root_path+'/Descriptor_ADMET_train.pkl','rb'))
data_test = pickle.load(open(root_path+'/Descriptor_ADMET_test.pkl','rb'))

## 筛除熵为0的列
feature_name = open(root_path+'/entropy_feat_threshold_0.txt','r').read().split('\n')[:-1] 
y_list = ['Caco-2','CYP3A4','hERG','HOB','MN']


data = data.sample(frac=1,random_state=2021).reset_index(drop=True)

X = data[feature_name]
y = data[y_list]

data_length = len(y)
idx = [i for i in range(data_length)]
split_num = int(data_length*0.2)
trn_data = X[:split_num]
# trn_data = X[idx[:split_num]]
trn_y = y[:split_num]
val_data = X[split_num:]
val_y = y[split_num:]

y_list = ['Caco-2','CYP3A4','hERG','HOB','MN']

##################### 线下验证 #####################
uauc_list = []
r_list = []

result_dict = {}

for item_y in y_list:
    print('=========', item_y, '=========')

    t = time.time()
    superpa = []
    for n_esti in range(1,200):
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=n_esti,
            num_leaves=255,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021,
            metric='None'
        )

        clf.fit(
            trn_data, trn_y[item_y],
            eval_set=[(val_data, val_y[item_y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )


        val_predcit = clf.predict_proba(val_data)[:, 1]
        val_result = []
        for item in val_predcit:
            if item > 0.5:
                val_result.append(1)
            else:
                val_result.append(0)
        acc = sum(val_result == val_y[item_y]) / len(val_result) 
        superpa.append(acc)
    result_dict[item_y] = superpa
    # plt.figure(figsize=[20,5])
    # plt.plot(range(1,200),superpa)
    # plt.show()
    # plt.savefig('./figure/quesiton3_lgb_n_estimators_'+item_y+'.jpg',dpi=200)
pickle.dump(result_dict,open('./question3_learning_curve_lgb.pkl','wb'))
        

    # val_uauc = uAUC(val_y[item_y], val_data[item_y + '_score'], val_x['userid'])

    # uauc_list.append(val_uauc)

    # print(val_uauc)

    # r_list.append(clf.best_iteration_)

    # print('runtime: {}\n'.format(time.time() - t))

##################### 全量训练 #####################

# ADMET_test = pd.read_excel("./code/data/ADMET.xlsx", sheet_name="test")

# #带有验证集验证效果
# r_dict = dict(zip(y_list, r_list))
# for item_y in y_list:
#     print('=========', item_y, '=========')

#     t = time.time()

#     clf = LGBMClassifier(
#         learning_rate=0.05,
#         n_estimators=1000,
#         num_leaves=255,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=2021,
#         metric='None'
#     )

#     clf.fit(
#         X, y[item_y],
#         eval_set=[(X, y[item_y])],
#         early_stopping_rounds=r_dict[item_y],
#         verbose=100
#     )

#     predcit_item_y = clf.predict_proba(data_test[feature_name])[:, 1]

#     print('runtime: {}\n'.format(time.time() - t))

# ADMET_test.to_csv('./quesiton3_test_predict_lgb.csv',index=False)