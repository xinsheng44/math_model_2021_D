# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:18:34 2021

@author: Hasee
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 01:29:30 2021

@author: Hasee
"""

import numpy as np
import matplotlib.pyplot as plt
fontsize = 20
font1 = {
     'family': 'Times New Roman',  # 字体
     'weight': 'normal',
    'size': fontsize  # 字体大小
}

font_l = {
     'family': 'Times New Roman',  # 字体
     'weight': 'normal',
    'size': 11  # 字体大小
}
font_M = {'family' : 'Times New Roman',
'weight' : 'heavy',
'size'   : 20}

P_or = [55.54]
P_wo_sge = [55.41]
P_wo_pos = [54.65]

Mrr_or = [19.68]
Mrr_wo_sge = [19.46]
Mrr_wo_pos = [18.22]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,1)
axes.bar(x, P_or, width=width, label='P@20 of DGS-MGCN ', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, P_wo_sge, width=width, label='P@20 of DGS-MGCN-w/o SGE', color="black",
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, P_wo_pos, width=width, label='P@20 of DGS-MGCN-w/o POS', color="green",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, Mrr_or, width=width, label='MRR@20 of DGS-MGCN', hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Mrr_wo_sge, width=width, label='MRR@20 of DGS-MGCN-w/o SGE', hatch='///', color="white",
         edgecolor='black')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Mrr_wo_pos, width=width, label='MRR@20 of DGS-MGCN-w/o POS', hatch='///', color="white",
         edgecolor='green')



ax2.set_ylim(15,21.5)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(53,56.5)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小
plt.xticks([])
plt.title('P@20 and MRR@20 on Digineica',font_M)
#plt.savefig('session_length.pdf')
#plt.show()


Y_P_or = [72.62]
Y_P_wo_sge = [72.34]
Y_P_wo_pos = [71.98]

Y_Mrr_or = [32.49]
Y_Mrr_wo_sge = [32.42]
Y_Mrr_wo_pos = [30.96]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,2)
axes.bar(x, Y_P_or, width=width, label='P@20 of of DGS-MGCN', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, Y_P_wo_sge, width=width, label='P@20 of DGS-MGCN-w/o SGE', color="black",
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, Y_P_wo_pos, width=width, label='P@20 of DGS-MGCN-w/o POS', color="green",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, Y_Mrr_or, width=width, label='MRR@20 of DGS-MGCN', hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Y_Mrr_wo_sge, width=width, label='MRR@20 of DGS-MGCN-w/o SGE', hatch='///', color="white",
         edgecolor='black')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Y_Mrr_wo_pos, width=width, label='MRR@20 of DGS-MGCN-w/o POS', hatch='///', color="white",
         edgecolor='green')



ax2.set_ylim(30,33)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(70,73.5)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小
plt.xticks([])
plt.title('P@20 and MRR@20 on Yoochoose',font_M)
#plt.savefig('session_length.pdf')
#plt.show()







R_P_or = [66.61]
R_P_wo_sge = [66.54]
R_P_wo_pos = [65.12]

R_Mrr_or = [45.78]
R_Mrr_wo_sge = [45.64]
R_Mrr_wo_pos = [43.89]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,3)
axes.bar(x, R_P_or, width=width, label='P@20 of DGS-MGCN', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, R_P_wo_sge, width=width, label='P@20 of DGS-MGCN-w/o SGE', color="black",
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, R_P_wo_pos, width=width, label='P@20 of DGS-MGCN-w/o POS', color="green",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, R_Mrr_or, width=width, label='MRR@20 of DGS-MGCN', hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, R_Mrr_wo_sge, width=width, label='MRR@20 of DGS-MGCN-w/o SGE', hatch='///', color="white",
         edgecolor='black')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, R_Mrr_wo_pos, width=width, label='MRR@20 of DGS-MGCN-w/o POS', hatch='///', color="white",
         edgecolor='green')



ax2.set_ylim(42,47)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(64,67.5)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小
plt.xticks([])
plt.title('P@20 and MRR@20 on Retailrocket',font_M)
#plt.savefig('session_length.pdf')
#plt.show()
plt.subplots_adjust(wspace =0.3, hspace =0.2)
plt.savefig('DGS-MGCN_SGE_POS.pdf')





































