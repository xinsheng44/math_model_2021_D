
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
 

root_path = '/home/bigdata9/wxs/game/math_model/code'
label_coloumn = ['pIC50']

train_path = root_path+'/data/train_Molecular_ER.xlsx'

train_data = pd.read_excel(train_path,sheet_name='Sheet1')
columns_list = train_data.columns.values.tolist()
X = train_data[columns_list[1:len(columns_list)-2]]
y = train_data[columns_list[-1]]
cor_list = []
## lgb 20 importance features
feature_name = ['MDEC-13','maxHBd','ETA_Eta_B_RC','ATSc5','SPC-6','ETA_Shape_Y','MLFER_E','MLFER_A','SC-5','ATSc4','XLogP','minHBint10','mindO','maxHaaCH','ALogP','ATSp5','VPC-5','VCH-7','minHsOH','maxwHBa']
## random forest 20 importance features  0.657
# feature_name = ['MDEC-23','minsssN','maxssO','maxHsOH','C1SP2','BCUTc-1l','minHsOH','LipoaffinityIndex','SHsOH','nHBAcc','minsOH','ATSc3','VC-5','MDEO-12','CrippenLogP','TopoPSA','BCUTc-1h','nC','XLogP','MLFER_A']
data = X[feature_name]
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data[data.columns].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white')
# sns.heatmap(data[data.columns].corr())
plt.savefig("./figure/lgb_feat_heatmap.jpg", dpi=500)

