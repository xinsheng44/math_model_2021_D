# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:06:42 2021

@author: Hasee
"""

import pandas as pd
import matplotlib.pyplot as plt
data = [54.78,54.81,54,56]
labels = ['1','2','3']
from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']

#plt.rcParams['figure.dpi'] = 300 #图片像素
plt.rcParams['figure.figsize'] = (4.0, 8.0)

 #cifar部分画在子图1中
Diginetica=[[54.78, 54.81, 54.35, 53.96, 53.65],
            [54.61, 54.21, 53.68, 53.12, 50.46],
            [19.27, 19.35, 19.16, 18.96, 18.80],
            [19.32, 19.14, 19.03, 18.87, 17.85]]
Tmall = [[38.40, 38.86, 39.10, 39.97, 36.18],
         [38.58, 38.76, 39.06, 39.36, 28.54],
         [18.33, 18.52, 18.61, 19.05, 17.65],
         [18.42, 18.51, 18.55, 18.81, 14.94]]
Nowplaying = [[23.64, 23.85, 23.67, 23.18, 20.80],
              [23.56, 23.58, 23.27, 20.58, 19.14],
              [7.72, 7.88, 7.46, 7.54, 7.28],
              [7.48, 7.42, 7.48, 7.28, 6.72]]
Retail = [[64.48, 65.02, 64.62, 62.42, 56.77],
          [64.93, 64.66, 63.54, 57.56, 55.87],
          [44.36, 44.42, 44.11, 42.68, 40.12],
          [44.24, 44.01, 43.26, 40.19, 39.25]]
colors=['red','orange','blue','green']
markers=['o','v','*','s']
P_20=['TE-GNN P@20','TE-GNN-HN P@20','TE-GNN MRR@20','TE-GNN-HN MRR@20']
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
font_M = {'family' : 'Times New Roman',
'weight' : 'heavy',
'size'   : 4,
}
x_bits = [1,2,3,4,5]
plt.subplots_adjust(wspace =0.2, hspace =0.35)
plt.subplot(4,2,1)
#开始绘制


for i in range(2):
    
    color= colors[i]
    plt.plot(x_bits,Diginetica[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
plt.yticks([i for i in range(48,56,2)],fontsize=4)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)


plt.subplot(4,2,2)
#开始绘制

for i in range(2,4):
    
    color= colors[i]
    plt.plot(x_bits,Diginetica[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
#plt.ylim(16,20)
plt.tick_params(length=1.2)
plt.yticks([i for i in range(16,21)],fontsize=4)       #y坐标范围
plt.title("MRR@20 on Diginetica",font_M)
plt.grid(linewidth =0.2)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)



plt.subplot(4,2,3)
#开始绘制


for i in range(2):
    
    color= colors[i]
    plt.plot(x_bits,Tmall[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
#plt.ylim(25,40)       #y坐标范围
plt.tick_params(length=1.2)
plt.yticks([i for i in range(25,41,4)],fontsize=4)
plt.title("P@20 on Tmall",font_M)
plt.grid(linewidth =0.2)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)


plt.subplot(4,2,4)
#开始绘制

for i in range(2,4):
    
    color= colors[i]
    plt.plot(x_bits,Tmall[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
plt.ylim(14,20)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.yticks([i for i in range(14,21,2)],fontsize=4)
plt.title("MRR@20 on Tmall",font_M)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)


##Nowplaying
plt.subplot(4,2,5)
#开始绘制


for i in range(2):
    
    color= colors[i]
    plt.plot(x_bits,Nowplaying[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
#plt.ylim(25,40)       #y坐标范围
plt.tick_params(length=1.2)
plt.yticks([i for i in range(16,26,2)],fontsize=4)
plt.title("P@20 on Nowplaying",font_M)
plt.grid(linewidth =0.2)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)


plt.subplot(4,2,6)
#开始绘制

for i in range(2,4):
    
    color= colors[i]
    plt.plot(x_bits,Nowplaying[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
#plt.ylim(14,20)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.yticks([i for i in range(5,9)],fontsize=4)
plt.title("MRR@20 on Nowplaying",font_M)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)


##retail
plt.subplot(4,2,7)
#开始绘制


for i in range(2):
    
    color= colors[i]
    plt.plot(x_bits,Retail[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
#plt.ylim(25,40)       #y坐标范围
plt.tick_params(length=1.2)
plt.yticks([i for i in range(50,67,2)],fontsize=4)
plt.title("P@20 on Retailrocket",font_M)
plt.grid(linewidth =0.2)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)


plt.subplot(4,2,8)
#开始绘制

for i in range(2,4):
    
    color= colors[i]
    plt.plot(x_bits,Retail[i],color=color,
         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
#plt.ylim(14,20)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.yticks([i for i in range(38,45)],fontsize=4)
plt.title("MRR@20 on Retailrocket",font_M)
plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)






plt.savefig('HighWay.pdf')
'''
#sun-----------------------------------
plt.subplot(4,2, 2)#sun部分画在子图1中
# 数据准备
y_early_students = [[0.75348, 0.8166, 0.82471, 0.82749],
                    [0.78058, 0.83077, 0.83574, 0.84841],
                    [0.8249, 0.86114, 0.8627, 0.87235],
                    [0.83924, 0.86368, 0.8674, 0.87599]
                    ]
y_ours_students = [[0.76123, 0.81592, 0.82724, 0.84166],
                   [0.78617, 0.83835, 0.84371, 0.85208],
                   [0.82594, 0.86230, 0.86595, 0.87225],
                   [0.83931, 0.86902, 0.87141, 0.87521]]

# 开始绘制
colors = ['red', 'orange', 'blue', 'green']
markers = ['o', 'v', '*', 's']
labels_early = ['student-1-early', 'student-2-early',
                'student-3-early', 'student-4-early']
labels_ours = ['student-1-ours', 'student-2-ours', 
               'student-3-ours', 'student-4-ours']
for i in range(4):
    color = colors[i]
    plt.plot(x_bits, y_early_students[i], color=color, 
             marker=markers[i], linestyle='--', label=labels_early[i])
    plt.plot(x_bits, y_ours_students[i], color=color, 
             marker=markers[i], linestyle='-', label=labels_ours[i])

plt.xticks(x_bits)  # 横轴只有这四个刻度
# plt.ylim(0.7, 0.9)       #y坐标范围
plt.title("SUN")
plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
plt.legend()
'''