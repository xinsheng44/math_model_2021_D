from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
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


if not os.path.exists('./Descriptor_ADMET_train.pkl'):
    Molecular_Descriptor_train = pd.read_excel("./code/data/Molecular_Descriptor.xlsx", sheet_name="training")
    ADMET_train = pd.read_excel("./code/data/ADMET.xlsx", sheet_name="training")
    train = Molecular_Descriptor_train.merge(ADMET_train,on=['SMILES'],how='left')
    pickle.dump(train,open('./Descriptor_ADMET_train.pkl','wb'))

if not os.path.exists('./Descriptor_ADMET_test.pkl'):
    Molecular_Descriptor_train = pd.read_excel("./code/data/Molecular_Descriptor.xlsx", sheet_name="test")
    ADMET_test = pd.read_excel("./code/data/ADMET.xlsx", sheet_name="test")
    test = Molecular_Descriptor_train.merge(ADMET_test,on=['SMILES'],how='left')
    pickle.dump(test,open('./Descriptor_ADMET_test.pkl','wb'))

data = pickle.load(open('./Descriptor_ADMET_train.pkl','rb'))
data_test = pickle.load(open('./Descriptor_ADMET_test.pkl','rb'))

## 筛除熵为0的列
feature_name = open('./entropy_feat_threshold_0.txt','r').read().split('\n')[:-1] 
y_list = ['Caco-2','CYP3A4','hERG','HOB','MN']


data = data.sample(frac=1,random_state=2021).reset_index(drop=True)

X = data[feature_name]
y = data[y_list[0]]


# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],voting='soft', weights=[2, 1, 2])

clf1 = clf1.fit(X, y)
clf2 = clf2.fit(X, y)
clf3 = clf3.fit(X, y)
eclf = eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
plt.savefig('./figure/question3_vote.py',dpi=200)