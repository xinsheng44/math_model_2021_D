# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:51:23 2021

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
    'size': 9  # 字体大小
}
font_M = {'family' : 'Times New Roman',
'weight' : 'heavy',
'size'   : 20}
P_or = [55.54]
P_bi_gru = [55.31]
P_str = [54.12]
P_seq = [54.01]

Mrr_or = [19.68]
Mrr_bi_gru = [19.52]
Mrr_str = [19.60]
Mrr_seq = [17.99]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 10
width = total_width / n
  # 字体大小
fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,1)
axes.bar(x, P_or, width=width, label='P@20 of DGS-MGNN ', color='darkgreen',hatch='x')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
    
#for i in range(len(x)):
#    x[i] = x[i] + width
#axes.bar(x, P_bi_gru, width=width, label='P@20 of DGS-MGCN-Bi GRU', color="springgreen",
#         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, P_str, width=width, label='P@20 of DGS-MGNN-STR', color="dodgerblue",hatch='*'
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, P_seq, width=width, label='P@20 of DGS-MGNN-SEQ', color="darkgrey",hatch='-'
         )


mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, Mrr_or, width=width, label='MRR@20 of DGS-MGNN', hatch='.', color="white",
         edgecolor='darkgreen')

#for i in range(len(MR_X)):
#    MR_X[i] = MR_X[i] + width
#
#ax2.bar(MR_X, Mrr_bi_gru, width=width, label='MRR@20 of DGS-MGCN-Bi GRU', hatch='//', color="white",
#         edgecolor='springgreen')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Mrr_str, width=width, label='MRR@20 of DGS-MGNN-STR', hatch='///', color="white",
         edgecolor='dodgerblue')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Mrr_seq, width=width, label='MRR@20 of DGS-MGNN-SEQ', hatch='o', color="white",
         edgecolor='darkgrey')


ax2.set_ylim(15,22.5)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l,handleheight = 3.1)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(53,57.0)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l,handleheight = 3.1)  # 图例位置、大小
plt.xticks([])
plt.title('(a) P@20 and MRR@20 on Digineica',font_M)
#plt.savefig('session_length.pdf')
#plt.show()


Y_P_or = [72.62]
Y_P_bi_gru = [72.37]
Y_P_str = [71.85]
Y_P_seq = [71.64]

Y_Mrr_or = [32.49]
Y_Mrr_bi_gru = [32.28]
Y_Mrr_str = [32.15]
Y_Mrr_seq = [31.96]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 10
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,2)
axes.bar(x, Y_P_or, width=width, label='P@20 of of DGS-MGNN', color='darkgreen',hatch='x')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
    
#for i in range(len(x)):
#    x[i] = x[i] + width
#axes.bar(x, Y_P_bi_gru, width=width, label='P@20 of DGS-MGCN-Bi GRU', color="springgreen",
#         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, Y_P_str, width=width, label='P@20 of DGS-MGNN-STR', color="dodgerblue",hatch='*'
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, Y_P_seq, width=width, label='P@20 of DGS-MGNN-SEQ', color="darkgrey",hatch='-'
         )


mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, Y_Mrr_or, width=width, label='MRR@20 of DGS-MGNN', hatch='.', color="white",
         edgecolor='darkgreen')

#for i in range(len(MR_X)):
#    MR_X[i] = MR_X[i] + width
#
#ax2.bar(MR_X, Y_Mrr_bi_gru, width=width, label='MRR@20 of DGS-MGCN-Bi GRU', hatch='//', color="white",
#         edgecolor='springgreen')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Y_Mrr_str, width=width, label='MRR@20 of DGS-MGNN-STR', hatch='///', color="white",
         edgecolor='dodgerblue')


for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Y_Mrr_seq, width=width, label='MRR@20 of DGS-MGNN-SEQ', hatch='o', color="white",
         edgecolor='darkgrey')

ax2.set_ylim(30,33.7)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l,handleheight = 3.1)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(70,73.9)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l,handleheight = 3.1)  # 图例位置、大小
plt.xticks([])
plt.title('(b) P@20 and MRR@20 on Yoochoose1/64',font_M)
#plt.savefig('session_length.pdf')
#plt.show()

###Yoo4
Y_P_or = [73.10]
Y_P_bi_gru = [72.37]
Y_P_str = [72.47]
Y_P_seq = [72.02]

Y_Mrr_or = [33.55]
Y_Mrr_bi_gru = [32.28]
Y_Mrr_str = [33.14]
Y_Mrr_seq = [32.98]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 10
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,3)
axes.bar(x, Y_P_or, width=width, label='P@20 of of DGS-MGNN', color='darkgreen',hatch='x')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
    
#for i in range(len(x)):
#    x[i] = x[i] + width
#axes.bar(x, Y_P_bi_gru, width=width, label='P@20 of DGS-MGCN-Bi GRU', color="springgreen",
#         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, Y_P_str, width=width, label='P@20 of DGS-MGNN-STR', color="dodgerblue",hatch='*'
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, Y_P_seq, width=width, label='P@20 of DGS-MGNN-SEQ', color="darkgrey",hatch='-'
         )


mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, Y_Mrr_or, width=width, label='MRR@20 of DGS-MGNN', hatch='.', color="white",
         edgecolor='darkgreen')

#for i in range(len(MR_X)):
#    MR_X[i] = MR_X[i] + width
#
#ax2.bar(MR_X, Y_Mrr_bi_gru, width=width, label='MRR@20 of DGS-MGCN-Bi GRU', hatch='//', color="white",
#         edgecolor='springgreen')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Y_Mrr_str, width=width, label='MRR@20 of DGS-MGNN-STR', hatch='///', color="white",
         edgecolor='dodgerblue')


for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, Y_Mrr_seq, width=width, label='MRR@20 of DGS-MGNN-SEQ', hatch='o', color="white",
         edgecolor='darkgrey')

ax2.set_ylim(30.5,35.7)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l,handleheight = 3.1)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(70,74.6)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l,handleheight = 3.1)  # 图例位置、大小
plt.xticks([])
plt.title('(c) P@20 and MRR@20 on Yoochoose1/4',font_M)

######









R_P_or = [66.61]
R_P_bi_gru = [66.34]
R_P_str = [65.52]
R_P_seq = [63.64]

R_Mrr_or = [45.78]
R_Mrr_bi_gru = [45.64]
R_Mrr_str = [45.60]
R_Mrr_seq = [43.12]



#MRR_short = [19.68,19.46,18.56]
#MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0]
total_width, n = 1.5, 10
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,4)
axes.bar(x, R_P_or, width=width, label='P@20 of DGS-MGNN', color='darkgreen',hatch='x')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=10)
  
#for i in range(len(x)):
#    x[i] = x[i] + width
#axes.bar(x, R_P_or, width=width, label='P@20 of DGS-MGCN', color="springgreen",
#         )

  
#for i in range(len(x)):
#    x[i] = x[i] + width
#axes.bar(x, R_P_bi_gru, width=width, label='P@20 of DGS-MGCN-Bi GRU', color="springgreen",
#         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, R_P_str, width=width, label='P@20 of DGS-MGNN-STR', color="dodgerblue",hatch='*'
         )

for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, R_P_seq, width=width, label='P@20 of DGS-MGNN-SEQ', color="darkgrey",hatch='-'
         )



mr_x = x[-1] + 1.2
MR_X = [mr_x]
ax2 = axes.twinx()
ax2.bar(MR_X, R_Mrr_or, width=width, label='MRR@20 of DGS-MGNN', hatch='.', color="white",
         edgecolor='darkgreen')

#for i in range(len(MR_X)):
#    MR_X[i] = MR_X[i] + width
#
#ax2.bar(MR_X, R_Mrr_bi_gru, width=width, label='MRR@20 of DGS-MGCN-Bi GRU', hatch='//', color="white",
#         edgecolor='springgreen')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, R_Mrr_str, width=width, label='MRR@20 of DGS-MGNN-STR', hatch='///', color="white",
         edgecolor='dodgerblue')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, R_Mrr_seq, width=width, label='MRR@20 of DGS-MGNN-SEQ', hatch='o', color="white",
         edgecolor='darkgrey')


ax2.set_ylim(41,48.3)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l,handleheight = 3.1)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

#plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(61,69.3)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l,handleheight = 3.1)  # 图例位置、大小
plt.xticks([])
plt.title('(d) P@20 and MRR@20 on Retailrocket',font_M)
#plt.savefig('session_length.pdf')
#plt.show()
plt.subplots_adjust(wspace =0.3, hspace =0.2)
plt.savefig('DGS-MGNN_STR_SEQ_1.pdf')