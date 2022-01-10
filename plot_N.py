# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 03:17:59 2021

@author: Hasee
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:06:42 2021

@author: Hasee
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = [54.78,54.81,54,56]
labels = ['1','2','3']
#font_M = {'family' : 'Times New Roman',
#'weight' : 'heavy',
#'size'   : 25}
from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']

#plt.rcParams['figure.dpi'] = 300 #图片像素
plt.rcParams['figure.figsize'] = (25, 25)

 #cifar部分画在子图1中
Diginetica=[[53.95, 55.12, 55.54, 55.61, 55.58, 55.53, 55.62, 55.35, 55.46, 55.61,55.59],
            [19.42, 19.56, 19.68, 19.63, 19.58, 19.64, 19.71, 19.70, 19.65, 19.66,19.68]]
Y64 = [[71.88, 72.46, 72.62, 72.57, 72.75, 72.70, 72.58, 72.66, 72.69, 72.59,72.60],
         [32.15, 32.33, 32.49, 32.55, 32.61, 32.58, 32.60, 32.56, 32.55, 32.58,32.55]]

Y4 = [[72.46, 72.88, 73.10, 73.20, 73.18, 73.15, 73.19, 73.21, 73.22, 73.16,73.18],
         [32.98, 33.33, 33.55, 33.58, 33.61, 33.56, 33.62, 33.57, 33.61, 33.56,33.59]]

retail = [[63.92,65.66,66.61,66.88,66.72,66.85,66.67,66.78,66.84,66.80,66.79],
          [45.21,45.69,45.78,45.92,45.98,45.85,45.94,45.89,45.91,45.88,45.96]]
#Nowplaying = [[23.64, 23.85, 23.67, 23.18, 20.80],
#              [23.56, 23.58, 23.27, 20.58, 19.14],
#              [7.72, 7.88, 7.46, 7.54, 7.28],
#              [7.48, 7.42, 7.48, 7.28, 6.72]]
#Retail = [[64.48, 65.02, 64.62, 62.42, 56.77],
#          [64.93, 64.66, 63.54, 57.56, 55.87],
#          [44.36, 44.42, 44.11, 42.68, 40.12],
#          [44.24, 44.01, 43.26, 40.19, 39.25]]
colors=['black','red','blue','green']
markers=['o','v','*','s']
P_20=['XXX P@20','XXX_w\o HighWay P@20','XXX MRR@20','XXX_w\o HighWay MRR@20']
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 22,
}
font_M = {'family' : 'Times New Roman',
'weight' : 'heavy',
'size'   : 25,
}
x_bits = [0,5,10,15,20,25,30,35,40,45,50]
plt.subplots_adjust(wspace =0.2, hspace =0.25)



plt.subplot(4,2,1)
#开始绘制


    
color= colors[0]
plt.plot(x_bits,Diginetica[0],color=color,
     marker=markers[0],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(53.5,57,0.6)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(a) P@20 on Digineica',font_M)


plt.subplot(4,2,2)
#开始绘制


    
color= colors[1]
plt.plot(x_bits,Diginetica[1],color=color,
     marker=markers[1],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(19,20,0.19)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(b) MRR@20 on Digineica',font_M)


#Tmall
plt.subplot(4,2,3)
#开始绘制


    
color= colors[0]
plt.plot(x_bits,Y64[0],color=color,
     marker=markers[0],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(71,73.2,0.4)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(c) P@20 on Yoochoose1/64',font_M)


plt.subplot(4,2,4)
#开始绘制


    
color= colors[1]
plt.plot(x_bits,Y64[1],color=color,
     marker=markers[1],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(31.5,33.1,0.3)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(d) MRR@20 on Yoochoose1/64',font_M)



###Y4

plt.subplot(4,2,5)
#开始绘制
color= colors[0]
plt.plot(x_bits,Y4[0],color=color,
     marker=markers[0],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(72,73.8,0.3)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(b) P@20 on Yoochoose1/4',font_M)




plt.subplot(4,2,6)
color= colors[1]
plt.plot(x_bits,Y4[1],color=color,
     marker=markers[1],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(32,34.3,0.4)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(c) MRR@20 on Yoochoose1/4',font_M)




##retail

plt.subplot(4,2,7)
#开始绘制
color= colors[0]
plt.plot(x_bits,retail[0],color=color,
     marker=markers[0],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(63.3,68.1,0.8)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(b) P@20 on Retailrocket',font_M)





plt.subplot(4,2,8)
#开始绘制
color= colors[1]
plt.plot(x_bits,retail[1],color=color,
     marker=markers[1],linewidth = 2,markersize=6,linestyle='-')
plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
plt.yticks([i for i in np.arange(44.5,46.5,0.3)],fontsize=15)       #y坐标范围
plt.tick_params(length=1.2)
plt.grid(linewidth =0.2)
plt.tick_params(length=5)
#plt.title("P@20 on Diginetica",font_M)
plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 3,
}
plt.legend(loc='lower left',prop=font3)
plt.title('(b) MRR@20 on Retailrocket',font_M)


    
plt.subplots_adjust(wspace=0.3,hspace=0.6)
plt.savefig('Nun_K.pdf',dp=200)
# plt.subplot(3,2,6)
# #开始绘制


    
# color= colors[1]
# plt.plot(x_bits,Y64[1],color=color,
#      marker=markers[1],linewidth = 2,markersize=6,linestyle='-')
# plt.xticks(x_bits,fontsize=15)  #横轴只有这四个刻度
# plt.yticks([i for i in np.arange(31.5,33.1,0.3)],fontsize=15)       #y坐标范围
# plt.tick_params(length=1.2)
# plt.grid(linewidth =0.2)
# plt.tick_params(length=5)
# #plt.title("P@20 on Diginetica",font_M)
# plt.xlabel("Number of global neighbors (K)",font2)  # 作用为横坐标轴添加标签  fontsize=12
# plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
# font3 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 3,
# }
# plt.legend(loc='lower left',prop=font3)
# plt.title('(d) MRR@20 on Yoochoose1/4',font_M)









#plt.savefig('Filed.pdf',dp=200)

#x = np.array([0.1,0.2,0.21,0.5])
#x = np.array([0.1,0.2,0.21,0.5,0.8,0.89])
#mins = x.min()
#maxs = x.max()
#degree = (maxs - mins)/6
#
#res = np.zeros(len(x))
#for i in range(1,7):
#    res[np.where((x >= mins + (i-1)*degree) & (x<= mins + i*degree))] = i
#    
#
#
#
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range+1e-10)

x = np.array([92.0, 82.0, 65.0, 32.0, 26.0, 9.0, 6.0, 0.0])
##x = np.array([0])
def softmax(x, N,axis=0):
#    N = 6
    N = len(x)
    z = normalization(x)
    print('norm x:',z)
#    tmp=np.max(x)
#    x-=tmp
#    x=np.exp(x)
#    tmp=np.sum(x)
#    s = x/tmp
    s = z
    mins = s.min()
    maxs = s.max()
    degree = (maxs - mins)/N
    res = np.zeros(len(x))
    for i in range(1,N+1):
        res[np.where((z >= mins + (i-1)*degree) & (z< 1e-10  + i*degree))] = i
    print('res:',res)
    return res.tolist()
#
#
#res = softmax(x,N=6)













##Nowplaying
#plt.subplot(4,2,5)
##开始绘制
#
#
#for i in range(2):
#    
#    color= colors[i]
#    plt.plot(x_bits,Nowplaying[i],color=color,
#         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
#plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
##plt.ylim(25,40)       #y坐标范围
#plt.tick_params(length=1.2)
#plt.yticks([i for i in range(16,26,2)],fontsize=4)
#plt.title("P@20 on Nowplaying",font_M)
#plt.grid(linewidth =0.2)
#plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
#plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
#font3 = {'family' : 'Times New Roman',
#'weight' : 'normal',
#'size'   : 3,
#}
#plt.legend(loc='lower left',prop=font3)
#
#
#plt.subplot(4,2,6)
##开始绘制
#
#for i in range(2,4):
#    
#    color= colors[i]
#    plt.plot(x_bits,Nowplaying[i],color=color,
#         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
#plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
##plt.ylim(14,20)       #y坐标范围
#plt.tick_params(length=1.2)
#plt.grid(linewidth =0.2)
#plt.yticks([i for i in range(5,9)],fontsize=4)
#plt.title("MRR@20 on Nowplaying",font_M)
#plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
#plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
#font3 = {'family' : 'Times New Roman',
#'weight' : 'normal',
#'size'   : 3,
#}
#plt.legend(loc='lower left',prop=font3)
#
#
###retail
#plt.subplot(4,2,7)
##开始绘制
#
#
#for i in range(2):
#    
#    color= colors[i]
#    plt.plot(x_bits,Retail[i],color=color,
#         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
#plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
##plt.ylim(25,40)       #y坐标范围
#plt.tick_params(length=1.2)
#plt.yticks([i for i in range(50,67,2)],fontsize=4)
#plt.title("P@20 on Retailrocket",font_M)
#plt.grid(linewidth =0.2)
#plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
#plt.ylabel("P@20(%)",font2)  # 作用为纵坐标轴添加标签
#font3 = {'family' : 'Times New Roman',
#'weight' : 'normal',
#'size'   : 3,
#}
#plt.legend(loc='lower left',prop=font3)
#
#
#plt.subplot(4,2,8)
##开始绘制
#
#for i in range(2,4):
#    
#    color= colors[i]
#    plt.plot(x_bits,Retail[i],color=color,
#         marker=markers[i],linewidth = 0.5,markersize=0.5,linestyle='--',label=P_20[i])
#plt.xticks(x_bits,fontsize=4)  #横轴只有这四个刻度
##plt.ylim(14,20)       #y坐标范围
#plt.tick_params(length=1.2)
#plt.grid(linewidth =0.2)
#plt.yticks([i for i in range(38,45)],fontsize=4)
#plt.title("MRR@20 on Retailrocket",font_M)
#plt.xlabel("Number of GNN layers",font2)  # 作用为横坐标轴添加标签  fontsize=12
#plt.ylabel("MRR@20(%)",font2)  # 作用为纵坐标轴添加标签
#font3 = {'family' : 'Times New Roman',
#'weight' : 'normal',
#'size'   : 3,
#}
#plt.legend(loc='lower left',prop=font3)






#plt.savefig('HighWay.pdf')
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