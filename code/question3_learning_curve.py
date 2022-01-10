import matplotlib.pyplot as plt
import random
from pylab import mpl
import pickle

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

y_list = ['Caco-2','CYP3A4','hERG','HOB','MN']

# 0.准备数据
x = range(1,201)
rf_para = pickle.load(open('./question3_rf_learning_curve.pkl','wb'))
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 5) for i in x]
y_guangzhou = [random.uniform(1, 5) for i in x]
y_xian = [random.uniform(1, 5) for i in x]

# 1.创建画布
# plt.figure(figsize=(20, 8), dpi=100)
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 8), dpi=100)


# 2.绘制图像
# plt.plot(x, y_shanghai, label="上海")
# plt.plot(x, y_beijing, color="r", linestyle="--", label="北京")
axes[0,0].plot(x, y_shanghai, label="Caco-2")
axes[0,1].plot(x, y_shanghai, label="Caco-2")
axes[1,0].plot(x, y_shanghai, label="CYP3A4")
axes[1,1].plot(x, y_shanghai, label="CYP3A4")
axes[2,0].plot(x, y_shanghai, label="hERG")
axes[2,1].plot(x, y_shanghai, label="hERG")
axes[3,0].plot(x, y_shanghai, label="HOB")
axes[3,1].plot(x, y_shanghai, label="HOB")
axes[0,0].plot(x, y_shanghai, label="MN")
axes[0,0].plot(x, y_shanghai, label="MN")
# axes[0,1].plot(x, y_beijing, color="r", linestyle="--", label="北京")
# axes[1,0].plot(x, y_guangzhou, color="b", linestyle="--", label="广州")
# axes[1,1].plot(x, y_xian, color="g", linestyle="--", label="西安")

# 2.1 添加x,y轴刻度
# 构造x,y轴刻度标签
x_ticks_label = ["11点{}分".format(i) for i in x]
y_ticks = range(40)

# 刻度显示
# plt.xticks(x[::5], x_ticks_label[::5])
# plt.yticks(y_ticks[::5])
axes[0,0].set_xticks(x[::5])
axes[0,0].set_yticks(y_ticks[::5])
axes[0,0].set_xticklabels(x_ticks_label[::5])
axes[0,1].set_xticks(x[::5])
axes[0,1].set_yticks(y_ticks[::5])
axes[0,1].set_xticklabels(x_ticks_label[::5])
axes[1,0].set_xticks(x[::5])
axes[1,0].set_yticks(y_ticks[::5])
axes[1,0].set_xticklabels(x_ticks_label[::5])
axes[1,1].set_xticks(x[::5])
axes[1,1].set_yticks(y_ticks[::5])
axes[1,1].set_xticklabels(x_ticks_label[::5])

# 2.2 添加网格显示
# plt.grid(True, linestyle="--", alpha=0.5)
axes[0,0].grid(True, linestyle="--", alpha=0.5)
axes[0,1].grid(True, linestyle="--", alpha=0.5)
axes[1,0].grid(True, linestyle="--", alpha=0.5)
axes[1,1].grid(True, linestyle="--", alpha=0.5)

# 2.3 添加描述信息
# plt.xlabel("时间")
# plt.ylabel("温度")
# plt.title("中午11点--12点某城市温度变化图", fontsize=20)
axes[0,0].set_xlabel("时间")
axes[0,0].set_ylabel("温度")
axes[0,0].set_title("中午11点--12点上海温度变化图", fontsize=20)
axes[0,1].set_xlabel("时间")
axes[0,1].set_ylabel("温度")
axes[0,1].set_title("中午11点--12点北京温度变化图", fontsize=20)
axes[1,0].set_xlabel("时间")
axes[1,0].set_ylabel("温度")
axes[1,0].set_title("中午11点--12点广州温度变化图", fontsize=20)
axes[1,1].set_xlabel("时间")
axes[1,1].set_ylabel("温度")
axes[1,1].set_title("中午11点--12点西安温度变化图", fontsize=20)

# # 2.4 图像保存
plt.savefig("./test.png")

# # 2.5 添加图例
# plt.legend(loc=0)
axes[0,0].legend(loc=0)
axes[0,1].legend(loc=0)
axes[1,0].legend(loc=0)
axes[1,1].legend(loc=0)


# 3.图像显示
plt.show()
