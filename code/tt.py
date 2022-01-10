from typing import Mapping
from lightgbm.callback import early_stopping
from lightgbm.sklearn import LGBMRegressor
import pandas as pd
import numpy as np
from seaborn import widgets
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import time
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
plt.style.use('ggplot')  #风格设置近似R这种的ggplot库
import seaborn as sns
sns.set_style('whitegrid')

np.random.seed(2021)
sns.set_theme()

root_path = '/home/bigdata9/wxs/game/math_model/code'
label_coloumn = ['pIC50']

train_path = root_path+'/data/train_Molecular_ER.xlsx'

max_features = 100
relationship_features = 500



#计算信息熵的方法
def calc_ent(X):
    """
    calculate shanno ent of x
    """
    result_df = pd.DataFrame()
    result_shanno = []

    columns = X.columns.values.tolist()
    for col in columns:
        ent = 0.0
        for item in X[col]:
            temp = np.array(X[col])
            p = float(temp[temp == item].shape[0]) / temp.shape[0]#计算每个元素出现的概率
#        print(p)
            logp = np.log2(p)
            ent -= p * logp
        result_shanno.append(ent)
    with open('./result_shano.txt','w') as f:
        for item in result_shanno:
            f.write(str(item))
            f.write('\n')


def draw(data):
    """
    import numpy as np; np.random.seed(0)
    import seaborn as sns; sns.set_theme()
    uniform_data = np.random.rand(10, 12)
    ax = sns.heatmap(uniform_data)
    """
    ax = sns.heatmap(data)


def lgb(X,y):
    params = {
    'learning_rate':0.05,
    'n_estimators':200,
    'num_leaves':255,
    'subsample':0.8,
    'colsample_bytree':1,
    'random_state':2021,
    'metric':'None'
    }
    # lgbc=LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    lgb = LGBMRegressor(**params)
    # ['ALogp2', 'ATSc3', 'ATSc4', 'ATSc5', 'ATSm4', 'BCUTw-1h', 'BCUTc-1l', 'BCUTp-1h', 'CrippenLogP', 'SHsOH', 'minaaCH', 'minaasC', 'maxwHBa', 'maxsOH', 'LipoaffinityIndex', 'MDEC-12', 'MDEC-33', 'MLFER_A', 'MLFER_BH', 'XLogP']
    embeded_lgb_selector = SelectFromModel(lgb, max_features=max_features).fit(X, y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
    # print(embeded_lgb_feature)
    # print(score)
    # print(embeded_lgb_selector.threshold_)
    return embeded_lgb_feature


def lgb_fea(X,y):
    params = {
    'learning_rate':0.05,
    'n_estimators':200,
    'num_leaves':255,
    'subsample':0.8,
    'colsample_bytree':1,
    'random_state':2021,
    'metric':'None'
    }
    model = LGBMRegressor(**params)
    model.fit(X,y)
    features = X.columns
    importances = model.feature_importances_
    importances_df = pd.DataFrame()

    idx = np.argsort(np.abs(importances))[-max_features:]
    feat_score = importances[idx].tolist()
    feat = features[idx]

    
    
    print(feat)
    print(feat_score)
    return feat

def cor_selector(X, y):
    # 皮尔逊相关系数
    ##从小到大
    cor_list = []    
    # feature_name = X.columns.tolist()
    # feature_name = open('./entropy_feat_threshold_0.txt','r').read().split('\n')[:-1]   
    feature_name = ['MDEC-23', 'LipoaffinityIndex', 'BCUTc-1l', 'ATSc5', 'XLogP',
       'ATSc4', 'minHsOH', 'minsssN', 'ATSc3', 'VC-5', 'SPC-6', 'MLFER_A',
       'maxHsOH', 'ALogP', 'SdssC', 'hmin', 'minssCH2', 'MDEC-33',
       'BCUTc-1h', 'minHBint10']
    # calculate the correlation with y for each feature    
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0,1]     
        cor_list.append(cor)   
    # replace NaN with 0    
    cor_list = [0 if np.isnan(i) else i for i in cor_list]   

    result_pierxun = pd.DataFrame([cor_list],columns=feature_name)
    result_pierxun.to_csv('./feature_entropy_pierxun_score.csv',index=False)


def draw_pierxun(X):
    cor_list = []
    feat = ['minHBint9', 'C1SP2', 'minaaN', 'gmax', 'nT5Ring', 'ALogp2', 'SHBa', 'C2SP1', 'minssssNp', 'SdsCH', 'maxssssNp', 'nB', 'VP-7', 'gmin', 'nAtomLAC', 'MDEC-22', 'BCUTp-1l', 'VP-3', 'VP-5', 'VP-6']
    data = X[feat]
    colormap = plt.cm.viridis
    plt.figure(figsize=(12,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(data[data.columns].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white')
    plt.savefig("./figure/feat_heatmap_pierxun_20.jpg", dpi=500)
           

 
def spearman(train_data,columns):
    ## spearman系数
    ##从小到大
    # data = train_data[columns[1:len(columns)-2]+[columns[-1]]] 


    feature_name = open('./entropy_feat_threshold_0.txt','r').read().split('\n')[:-1]   
    data = train_data[feature_name + label_coloumn]
    spearman = data.corr('spearman')
    result_spearman = spearman[columns[-1]][:].tolist()
    result_spearman = [0 if np.isnan(i) else i for i in result_spearman]

    result_pierxun = pd.DataFrame([result_spearman[:-1]],columns=feature_name)
    result_pierxun.to_csv('./feature_entropy_spearman_score.csv',index=False)

    max_spearman = np.array(result_spearman)[np.argsort(np.abs(result_spearman))[-max_features-1:-1]].tolist()
    max_feature = data.iloc[:,np.argsort(np.abs(result_spearman))[-max_features-1:-1]].columns.tolist() 
    # print(max_feature)
    # print(max_spearman)
    # with open('./feature_importance_spearman.txt','w') as f:
    #     for item in result_spearman[:-1]:
    #         f.write(str(item))
    #         f.write('\n')
    return max_feature



def randForest(X,y):
    embeded_rf_selector = SelectFromModel(RandomForestRegressor(n_estimators=100), max_features=max_features).fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    return embeded_rf_feature


def merge():
    Molecular_Descriptor_train = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/Molecular_Descriptor.xlsx", sheet_name="training")
    ER_alpha_activity_train = pd.read_excel("/home/bigdata9/wxs/game/math_model/code/data/ERα_activity.xlsx", sheet_name="training")
    train = Molecular_Descriptor_train.merge(ER_alpha_activity_train,on=['SMILES'],how='left')
    train.to_excel(train_path,index=False)


def caculate_score():

    model_name = ['pierxun','spearman','randomForest','lgb']
    weights = [1,1,1,1]
    result_score = []
    feature_name = open('./entropy_feat_threshold_0.txt','r').read().split('\n')[:-1]
    for idx,model in enumerate(model_name):
        path = 'feature_entropy_'+ model+'_score.csv'
        df = pd.read_csv(path)
        score_list = df.values.tolist()[0]
        idx_score = np.argsort(np.abs(score_list))[-max_features:]
        feat_max = np.array(feature_name)[idx_score]
        score_max_features = df.iloc[:,idx_score].values.tolist()[0]
        
        mean = np.average(score_max_features)
        std = np.std(score_max_features)
        score_list = (score_max_features - mean) / std
        std_df = pd.DataFrame([score_list],columns=feat_max)
        std_df.to_csv('./feat_score_std_'+model+'_maxFeatuers_'+str(max_features)+'.csv',index=False)
        



def select_feature(X):

    ## 通过每列熵大小筛选列值
    feat = X.columns.values.tolist()
    threshold = 0
    
    data_shano = open('./result_shano.txt','r').readlines()
    data_shano = [float(item) for item in data_shano]
    select_list = [np.array(data_shano) > threshold]
    
    with open('./entropy_feat_threshold_'+str(threshold)+'.txt','w') as f:
        for item in np.array(feat)[select_list]:
            f.write(item)
            f.write('\n')
    f.close()
    t = sum(select_list)
    print('feat_nums:'+str(np.sum(t)))



def select_byBiRelation():

    pierxun_df = pd.read_csv('feature_entropy_pierxun_score.csv')
    spearman_df = pd.read_csv('feature_entropy_spearman_score.csv')
    feat_name = pierxun_df.columns.values.tolist()
    result = []
    columns_ = []
    t = 0.05  ## 421
    divide_score = pierxun_df[feat_name]+spearman_df[feat_name] / 2
    divide_score.to_csv('./pierxun_spearman_mean.csv')



if __name__ == '__main__':
    statis_data()
    merge()
    train_data = pd.read_excel(train_path,sheet_name='Sheet1')
    columns_list = train_data.columns.values.tolist()
    X = train_data[columns_list[1:len(columns_list)-2]]
    y = train_data[columns_list[-1]]

    

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    select_feature(X)
    draw_pierxun(X)
    calc_ent(X)
    lgb_fea(X,y)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    lgb_feature = lgb(X,y)
    pierxun_feature = cor_selector(X, y)
    
    spearman_feature = spearman(train_data,columns_list)
    caculate_score()
    select_byBiRelation()
    rf_feature = randForest(X,y)


    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
 