from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
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


root_path = '/home/bigdata9/wxs/game/math_model'
if not os.path.exists(root_path+'/Descriptor_ADMET_train.pkl'):
    Molecular_Descriptor_train = pd.read_excel(root_path+"/code/data/Molecular_Descriptor.xlsx", sheet_name="training")
    ADMET_train = pd.read_excel(root_path+"/code/data/ADMET.xlsx", sheet_name="training")
    train = Molecular_Descriptor_train.merge(ADMET_train,on=['SMILES'],how='left')
    pickle.dump(train,open(root_path+'/Descriptor_ADMET_train.pkl','wb'))

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
## 421
t = 0.05  ## 421
# data_421 = pd.read_csv('./feature_entropy_pierxun_spearman_divide_score_select_'+str(t)+'.csv')
# feature_name = data_421.columns.values.tolist()


data = data.sample(frac=1,random_state=2021).reset_index(drop=True)

X = data[feature_name]
y = data[y_list]

data_length = len(y)
idx = [i for i in range(data_length)]
split_num = int(data_length*0.8)
trn_data = X[:split_num]
# trn_data = X[idx[:split_num]]
trn_y = y[:split_num]
val_data = X[split_num:]
val_y = y[split_num:]


# ADMET_test_class = pd.read_excel(root_path+"/code/data/ADMET.xlsx", sheet_name="test")
# ADMET_test_prob = pd.read_excel(root_path+"/code/data/ADMET.xlsx", sheet_name="test")
################### 开始训练 ##################
uauc_list = []
r_list = []

for item_y in y_list:
    print('=========', item_y, '=========')

    t = time.time()


    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=200,random_state=2021)
    # Fit on training data
    model.fit(trn_data, trn_y[item_y])
    


    val_predict =  model.predict(val_data)
    val_result = []

    acc = sum(val_predict==val_y[item_y]) / len(val_y[item_y])
    print('acc:',acc)
    # ADMET_test_prob[item_y] =  model.predict_proba(data_test[feature_name])

    print('runtime: {}\n'.format(time.time() - t))

print('=========', item_y, '=========')

t = time.time()


# # Create the model with 100 trees
# model = RandomForestClassifier(max_depth=2,random_state=2021)
# # Fit on training data
# model.fit(trn_data, trn_y)



# val_predict =  model.predict(val_data)
# val_result = []
# for idx,item_y in enumerate(y_list):
#     acc = sum(val_predict[:,idx]==val_y[item_y]) / len(val_y[item_y])
#     print('acc:',acc)
# # ADMET_test_prob[item_y] =  model.predict_proba(data_test[feature_name])

# print('runtime: {}\n'.format(time.time() - t))
# result_dict = {}
# y_list = ['Caco-2','CYP3A4','hERG','HOB','MN']
# for item_y in y_list:
#     print('=========', item_y, '=========')
#     #6. n_estimators的学习曲线
#     superpa = []
#     for i in range(200):
#         rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1) #这里就是进行了200次的随机森林计算，每次的n_estimator设置不一样
#         rfc.fit(trn_data,trn_y[item_y])
#         score = rfc.score(val_data,val_y[item_y])
#         superpa.append(score)
#     print(max(superpa),superpa.index(max(superpa)))
#     # plt.figure(figsize=[20,5])
#     # plt.plot(range(1,201),superpa)
#     # plt.show()
#     # plt.savefig('./figure/quesiton3_rf_n_estimators'+item_y+'.jpg',dpi=200)
#     result_dict[item_y] = superpa
# pickle.dump(result_dict,open('./question3_rf_learning_curve.pkl','rb'))



# list.index(object) >>>返回对象object在列表list中的索引 68是i值，但是n_estimators=i+1，所以最大准确率对应的n_estimators是69.

# ADMET_test_class.to_csv('./quesiton3_test_predict_rf_calss.csv',index=False)
# ADMET_test_prob.to_csv('./quesiton3_test_predict_rf_prob.csv',index=False)
##################### 全量训练 #####################

# ADMET_test = pd.read_excel("./code/data/ADMET.xlsx", sheet_name="test")

# #带有验证集验证效果
# r_dict = dict(zip(y_list, r_list))
# for item_y in y_list:
#     print('=========', item_y, '=========')

#     t = time.time()

#     clf = LGBMClassifier(
#         learning_rate=0.05,
#         n_estimators=r_dict[item_y],
#         num_leaves=63,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=2021
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