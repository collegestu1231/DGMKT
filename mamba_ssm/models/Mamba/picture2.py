import matplotlib.pyplot as plt
import numpy as np

# 设置字体
# plt.rcParams['font.family'] = 'SimHei'

# 数据
models = [r'$DGMKT\,\mathbf{w/o}\,HD&DG$', r'$DGMKT\,\mathbf{w/o}\,DG$', r'$DGMKT\,\mathbf{w/o}\,HG$', r'$DGMKT$']

datasets = ['statics2011', 'kddcup2010', 'Assistments2009', 'Assistments2017']  # 数据集名称
scores = {
    r'$DGMKT\,\mathbf{w/o}\,HD&DG$': [0.8085, 0.7865, 0.8086, 0.7173],
    r'$DGMKT\,\mathbf{w/o}\,DG$': [0.8235, 0.7922, 0.8132, 0.7252],
    r'$DGMKT\,\mathbf{w/o}\,HG$': [0.8159, 0.7838, 0.8124, 0.7236],
    r'$DGMKT$': [0.8261, 0.7986, 0.8180, 0.7339],
}


# 数据集数量
n_datasets = len(datasets)

# 创建图形
fig, ax = plt.subplots(figsize=(18, 7))

# 设置条形图宽度和位置
width = 0.20
x = np.arange(len(models))  # X轴的位置

# 自定义颜色，确保更清晰的展示
light_colors = ['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4']  # 使用更柔和的颜色
light_colors = ['#C6E7FF','#D4F6FF','#FBFBFB','#FFDDAE']
# light_colors = ['#89A8B2','#B3C8CF','#E5E1DA','#F1F0E8']
light_colors = ['#8294C4','#ACB1D6','#DBDFEA','#FFEAD2']


# 绘制条形图
for i, dataset in enumerate(datasets):
    ax.bar(x + i * width, [scores[model][i] for model in models], width, label=dataset, color=light_colors[i],edgecolor='black',linewidth=2)

# 设置标签和标题
ax.set_xticks(x + width * 1.5)  # 设置X轴的位置，使条形图居中
ax.set_xticklabels(models, fontsize=16)

ax.set_xlabel('Models', fontsize=32)
ax.set_ylabel('AUC', fontsize=32)
ax.set_title('', fontsize=16)

# 在条形上添加数值标签，保留三位小数
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        ax.text(x[j] + i * width, scores[model][i] + 0.001, f'{scores[model][i]:.3f}',
                ha='center', va='bottom', fontsize=16)

# 设置y轴的显示范围
ax.set_ylim(0.7, 0.88)  # 适当扩大y轴范围，避免图例重叠

# 添加图例
ax.legend(title='', fontsize=20)
ax.tick_params(axis='x', labelsize=28)  # X轴标签字体大小

# 调整y轴标签字体大小
ax.tick_params(axis='y', labelsize=24)  # Y轴标签字体大小

# 优化布局，避免元素重叠
plt.tight_layout()

# 显示图形
plt.show()
