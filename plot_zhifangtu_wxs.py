
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def draw_pic(data):
    # plt.rcParams['font.sans-serif'] = ['SimHei']				# 解决中文无法显示的问题
    # plt.rcParams['axes.unicode_minus'] = False			# 用来正常显示负号												
    plt.hist(sorted(data), bins=10)								# bins表示分为5条直方，可以根据需求修改
    # plt.xlabel('区间范围')
    plt.ylabel('frequency')
    plt.show()
    plt.savefig('./figure/lgb_zhifangtu.jpg',dpi=100)

if __name__ == '__main__':
    data = pd.read_csv('./feature_entropy_lgb_score.csv')
    data = np.array(data.values.tolist()[0]) / sum(data.values.tolist()[0])
    print(data)
    draw_pic(data.tolist())
