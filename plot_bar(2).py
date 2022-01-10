# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 01:29:30 2021

@author: Hasee
"""

import numpy as np
import matplotlib.pyplot as plt
fontsize = 20
name_list = ['TAGNN', 'GCE-GNN','TE-GNN','TAGNN', 'GCE-GNN','TE-GNN']
font1 = {
     'family': 'Times New Roman',  # 字体
     'weight': 'normal',
    'size': fontsize  # 字体大小
}

font_l = {
     'family': 'Times New Roman',  # 字体
     'weight': 'normal',
    'size': 12  # 字体大小
}
font_M = {'family' : 'Times New Roman',
'weight' : 'heavy',
'size'   : 20}

P_short = [53.25, 54.76, 55.25]
P_long = [50.25, 51.88, 52.32]
MRR_short = [18.85,19.58,20.11]
MRR_long = [16.15,17.04,17.31]
x_width = 0.8  # 调节横宽度
x = [0, x_width,x_width*2]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,1)
axes.bar(x, P_long, width=width, label='P@20 of long session', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=11.5)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, P_short, width=width, label='P@20 of short session', color="black",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x,mr_x+x_width,mr_x+x_width*2]
ax2 = axes.twinx()
ax2.bar(MR_X, MRR_long, width=width, label='MRR@20 of long session', hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, MRR_short, width=width, label='MRR@20 of short session', hatch='///', color="white",
         edgecolor='green')
ax2.set_ylim(15,22)

ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X
#ax2.set_xticklabels(MR_X,

plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list)  # 调节横坐标不居中
#plt.yticks(size = 20) :L"K
#plt.tick_params(labelsize=20)
#plt.xticks([0,1,2,3,4,5],name_list,fontsize=20)
axes.set_ylim(45,57)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小

plt.title('P@20 and MRR@20 on Digineica',font_M)
#plt.savefig('session_length.pdf')
#plt.show()

##Tmall
T_P_short = [23.25, 26.54, 30.50]
T_P_long = [35.82, 35.66 ,41.12]
T_MRR_short = [12.18,12.95 , 15.12]
T_MRR_long = [16.78, 15.94, 19.67]
x_width = 0.8  # 调节横宽度
x = [0, x_width,x_width*2]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,2)
axes.bar(x, T_P_long, width=width, label='P@20 of long session', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=11.5)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, T_P_short, width=width, label='P@20 of short session', tick_label=name_list[:3], color="black",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x,mr_x+x_width,mr_x+x_width*2]
ax2 = axes.twinx()
ax2.bar(MR_X, T_MRR_long, width=width, label='MRR@20 of long session', tick_label=name_list[3:], hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, T_MRR_short, width=width, label='MRR@20 of short session', hatch='///', color="white",
         edgecolor='green')
ax2.set_ylim(10,22)
ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X

plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list,fontsize=20)  # 调节横坐标不居中

axes.set_ylim(20,45)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小

plt.title('P@20 and MRR@20 on Tmall',font_M)




#Nowplaying
T_P_short = [17.52, 21.27, 22.80]
T_P_long = [20.60, 24.46, 25.74]
T_MRR_short = [6.57,7.72 ,7.23]
T_MRR_long = [7.91, 9.12, 8.23]
x_width = 0.8  # 调节横宽度
x = [0, x_width,x_width*2]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,3)
axes.bar(x, T_P_long, width=width, label='P@20 of long session', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=11.5)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, T_P_short, width=width, label='P@20 of short session', tick_label=name_list[:3], color="black",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x,mr_x+x_width,mr_x+x_width*2]
ax2 = axes.twinx()
ax2.bar(MR_X, T_MRR_long, width=width, label='MRR@20 of long session', tick_label=name_list[3:], hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, T_MRR_short, width=width, label='MRR@20 of short session', hatch='///', color="white",
         edgecolor='green')
ax2.set_ylim(0,15)
ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X

plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list,fontsize=20)  # 调节横坐标不居中

axes.set_ylim(15,28)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小

plt.title('P@20 and MRR@20 on Nowplaying',font_M)




##retail
T_P_short = [58.51, 64.42, 68.12]
T_P_long = [54.12, 60.59, 62.10]
T_MRR_short = [40.36,42.86 , 46.23]
T_MRR_long = [34.42, 34.21 ,40.12]
x_width = 0.8  # 调节横宽度
x = [0, x_width,x_width*2]
total_width, n = 1.5, 6
width = total_width / n
  # 字体大小
#fig = plt.figure(figsize=(16, 16), dpi=120, facecolor='white', edgecolor="white")  # figsize长、宽
axes = plt.subplot(2,2,4)
axes.bar(x, T_P_long, width=width, label='P@20 of long session', color='red')  # hatch：填充图案，edgecolor：填充图案颜色，color：柱形图颜色
plt.tick_params(labelsize=11.5)
    
for i in range(len(x)):
    x[i] = x[i] + width
axes.bar(x, T_P_short, width=width, label='P@20 of short session', tick_label=name_list[:3], color="black",
         )

mr_x = x[-1] + 1.2
MR_X = [mr_x,mr_x+x_width,mr_x+x_width*2]
ax2 = axes.twinx()
ax2.bar(MR_X, T_MRR_long, width=width, label='MRR@20 of long session', tick_label=name_list[3:], hatch='///', color="white",
         edgecolor='red')

for i in range(len(MR_X)):
    MR_X[i] = MR_X[i] + width

ax2.bar(MR_X, T_MRR_short, width=width, label='MRR@20 of short session', hatch='///', color="white",
         edgecolor='green')

ax2.set_ylim(30,50)
ax2.set_ylabel('MRR@20(%)', font1)
ax2.legend(loc="upper right", prop=font_l)
tt = x + MR_X

plt.xticks(np.asarray(tt) + total_width / 12 -width,name_list,fontsize=20)  # 调节横坐标不居中
#axes.set_xticks(size=16)
axes.set_ylim(50,72)
axes.set_ylabel('P@20(%)', font1)
axes.legend(loc="upper left", prop=font_l)  # 图例位置、大小

plt.title('P@20 and MRR@20 on Retailrocket',font_M)
plt.subplots_adjust(wspace =0.3, hspace =0.2)
#plt.xticks(fontproperties = 'Times New Roman', size = 1)
plt.savefig('session_length.pdf')

#
#x = [1,2,3,4]
#y =[5,6,7,8]
#z = 
#plt.bar(x,y)
#plt.xticks(fontsize=10)


