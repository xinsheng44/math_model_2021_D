import os
import matplotlib.pyplot as plt
import deepchem
import pandas as pd
import numpy as np

root_path = '/home/bigdata9/wxs/game/math_model/code'

def learn_curve():
     ##### 画图 #####
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}   # 设置字体大小
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}   # 标题大小
    plt.figure(figsize=(6, 8))     # 设置画布大小
    # 20，40，60数据的横坐标
    x = [40,80,120,160,200]

    # 图一
    ax1 = plt.subplot(2, 1, 1)
    plt.xlim(5, None)   # x 轴刻度从0开始
    plt.ylim(0.5, None)   # y 轴刻度从0开始
    plt.tick_params(direction="in", labelsize=16)   # direction设置刻度线在内显示，labelsize设置x和y轴字体大小

    # 取数据
    with open(root_path+'/lgb_eval_loss.txt') as f:
        temp = f.readlines()
        loss_list = [float(item) for item in temp]
  


    # 画线
    plt.plot([i for i in range(len(loss_list))], loss_list, color='red')
    # plt.plot([i for i in range(len(loss_list))], loss_list, label='teacher-ori-graph', color='red', marker='.')
    # plt.plot(x, tea_knn_graph, label='teacher-knn-graph', color='green', marker='.')
    # plt.plot(x, cpf_ind, label='CPF-ind', color='blue', marker='.')
    # plt.plot(x, cpf_tra, label='CPF-tra', color='violet', marker='.')

    # 设置y刻度

    plt.yticks([1.0,1.5,2.0,2.5,3.0,3.5])


    # 设置x刻度
    plt.xticks([40,80,120,160,200])
    # plt.xticks([1,2,3,4],['0','50','150','200'])


    # 设置字体
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置x，y轴表示
    plt.xlabel('# iteration nums', font1)
    plt.ylabel('MSE ', font1)
    plt.legend(prop=font1, loc=4)   # 图例 右下角的20，40，60
    # plt.grid(axis="y")    # y轴的横刻度线

    # 标题
    # dataset_dic = {"acm": "ACM", "citeseer": "CiteSeer", "uai": "UAI2010",
    #              "blogcatalog": "BlogCatalog", "flickr": "Flickr"}
    # plt.title(dataset_dic[dataset], font2)

    if not os.path.exists('./figure'):
        os.makedirs('./figure')
    plt.savefig("./figure/lgb.jpg", dpi=500)
    plt.show()


def data_distribution():

    # Import Data
    path = './result_shano.txt'
    str_list = open(path,'r').readlines()
    score_list = [np.abs(float(item)) for item in str_list]


    plt.hist(np.array(score_list), 5)
    plt.show()
    plt.savefig('./figure/entropy_distribution.jpg',dpi=500)


if __name__ == '__main__':
    # learn_curve()
    data_distribution()






