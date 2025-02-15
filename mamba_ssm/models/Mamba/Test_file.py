from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
def edit_distance(list1, list2):
    len1, len2 = len(list1), len(list2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 初始化边界情况
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 填充dp数组
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[len1][len2]


def find_closest_and_farthest(lists):
    # 选取第一个子列表作为基准列表
    base_list = lists[0]

    # 用于记录每个子列表与基准列表的编辑距离
    distances = []

    # 遍历从第二个子列表开始的每个子列表，计算与基准列表的编辑距离
    for i in range(1, len(lists)):
        distance = edit_distance(base_list, lists[i])  # 假设 edit_distance 已定义
        distances.append((i, distance))  # 保存索引和距离

    # 根据编辑距离对列表进行排序
    distances.sort(key=lambda x: x[1])  # 从小到大排序

    # 分成两份：距离小的放入 closest_indices，距离大的放入 farthest_indices
    midpoint = len(distances) // 2
    closest_indices = [index for index, _ in distances[:midpoint]]
    farthest_indices = [index for index, _ in distances[midpoint:]]

    return closest_indices, farthest_indices




def remove_negative_ones(array):
    # 创建一个新的列表，用于存储去除 -1 后的元素
    cleaned_list = []

    # 遍历原始数组的每一个子列表
    for sublist in array:
        # 去除子列表中的所有 -1，并添加到新的列表中
        cleaned_sublist = [item for item in sublist if item != -1]
        cleaned_list.append(cleaned_sublist)

    return cleaned_list

def weighted_jaccard_similarity(list1, list2):
    # 使用 Counter 计算每个列表中元素的出现次数
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    # 找到两个列表中所有的唯一元素
    union_keys = set(counter1.keys()).union(set(counter2.keys()))

    # 计算加权交集和加权并集
    intersection_weight = 0
    union_weight = 0

    for key in union_keys:
        count1 = counter1.get(key, 0)
        count2 = counter2.get(key, 0)
        intersection_weight += min(count1, count2)
        union_weight += max(count1, count2)

    # 计算加权 Jaccard 相似度
    return intersection_weight / union_weight if union_weight != 0 else 0


def find_closest_and_farthest_weight_jacc(lists):
    # 选取第一个子列表作为基准列表
    base_list = lists[0]

    # 用于记录每个子列表与基准列表的 Jaccard 相似度
    similarities = []

    # 遍历从第二个子列表开始的每个子列表，计算与基准列表的加权 Jaccard 相似度
    for i in range(1, len(lists)):
        similarity = weighted_jaccard_similarity(base_list, lists[i])
        similarities.append((i, similarity))

    # 根据相似度对列表进行排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 分成两份：相似度大的放入 closest_indexs，相似度小的放入 farthest_indexs
    midpoint = len(similarities) // 2
    closest_indices = [index for index, _ in similarities[:midpoint]]
    farthest_indices = [index for index, _ in similarities[midpoint:]]

    return closest_indices, farthest_indices





def plot_radar_chart(tensor_1, tensor_2, concept_list=None,step=None):
    # 检查输入是否是长度为5的torch.Tensor
    if not (isinstance(tensor_1, torch.Tensor) and isinstance(tensor_2, torch.Tensor)):
        raise ValueError("输入必须是torch.Tensor类型。")
    if not (tensor_1.size(0) == 5 and tensor_2.size(0) == 5):
        raise ValueError("输入的Tensor长度必须为5。")

    # 将Tensor转换为列表
    values_1 = tensor_1.tolist()
    values_2 = tensor_2.tolist()

    # 标签设置
    if concept_list is None:
        labels = ['concept1', 'concept2', 'concept3', 'concept4', 'concept5']
    else:
        if len(concept_list) != 5:
            raise ValueError("concept_list长度必须为5。")
        labels = [f'concept{concept}' for concept in concept_list]

    num_vars = len(labels)

    # 将数据循环封闭
    values_1 += values_1[:1]
    values_2 += values_2[:1]

    # 计算每个角的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 设置画布
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制第一个数据集
    ax.plot(angles, values_1, color='orange', linewidth=2, label='STEP = 0')

    # 绘制第二个数据集
    ax.plot(angles, values_2, color='blue', linewidth=2, label=f'STEP = {step} ')

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # 设置最大值刻度
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
    ax.set_ylim(0, 1)

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # 显示图形
    plt.show()

# 示例用法
# tensor_1 = torch.tensor([0.7, 0.8, 0.5, 0.6, 0.4])
# tensor_2 = torch.tensor([0.5, 0.9, 0.6, 0.4, 0.3])
# plot_radar_chart(tensor_1, tensor_2)
def find_unique_steps(tensor):
    # 创建一个空的集合来保存找到的不同的值
    unique_values = []
    unique_values1 = set()
    # 初始化一个计数器来计算步数
    steps = 0

    # 遍历张量中的每个元素
    for value in tensor:
        # 如果当前值不在集合中，添加到集合中
        if value.item() not in unique_values:
            unique_values.append(value.item())

        # 增加步数
        steps += 1
        # 当找到不同的5个值时，停止
        if len(unique_values) > 5:
            break

    return steps-2,unique_values[:-1]


# 示例用


import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
def plot_mastery_heatmap(tensor, concept_list=None):
    # 检查输入是否是 torch.Tensor 类型并转换为 NumPy 数组
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("输入必须是 torch.Tensor 类型。")
    data = tensor.cpu().numpy()

    # 设置固定的图像宽度和高度
    fixed_width = 36  # 设置合理的图像宽度（单位：英寸）
    fixed_height = 10  # 设置合理的图像高度（单位：英寸）
    plt.figure(figsize=(fixed_width, fixed_height))  # 固定图像的大小

    # 创建热力图
    ax = sns.heatmap(data.T, cmap="Blues", cbar_kws={'label': 'Mastery Degree'}, annot=False, linewidths=2.0, square=False)

    # 设置图表标题和标签
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")

    # 自定义Y轴的标签名
    if concept_list is None:
        yticks_labels = ['finding-percents', 'circle-graph', 'pattern-finding', 'unit-conversion', 'noskill']
    else:
        yticks_labels = [f'concept{concept}' for concept in concept_list]

    ax.set_yticklabels(yticks_labels, rotation=0, fontsize=24)

    # 显示X轴的时间步，确保X轴刻度在每个块的中心
    ax.set_xticks(np.arange(tensor.size(0)) + 0.5)  # 每个块的中心位置
    ax.set_xticklabels([str(i + 1) for i in range(tensor.size(0))], fontsize=24, rotation=0)

    # 设置colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # 设置colorbar的标签字体大小
    cbar.set_label('Mastery Degree', fontsize=20)
    cbar.set_ticks([0.1,  0.3, 0.5, 0.7])
    cbar.ax.set_yticklabels([f'0.1', '0.3', '0.5', '0.7'], fontsize=16)  # 设置刻度标签

    # 设置colorbar的高度和间距
    cbar.ax.set_aspect(10)  # 调整colorbar的高度比例

    # 调整cbar与热力图的间距
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 根据需要调整间距

    # 显示图表
    plt.show()

import pandas as pd

# 原始数据
data = {
    'finding-percents': [0.2305, 0.2437, 0.3584, 0.3268, 0.3136, 0.3042, 0.2976, 0.294, 0.2912, 0.2886,
                         0.5269, 0.5386, 0.5052, 0.3437, 0.6227, 0.3358, 0.1842, 0.3519, 0.265, 0.3843,
                         0.3624, 0.4505, 0.3458, 0.2413, 0.2598, 0.3259, 0.2373, 0.2338, 0.3027, 0.304,
                         0.3849, 0.3562, 0.4268, 0.4043, 0.4124, 0.4205],
    'circle-graph': [0.3421, 0.2188, 0.2683, 0.2534, 0.2243, 0.2064, 0.1999, 0.1935, 0.1879, 0.1834,
                     0.3798, 0.4386, 0.4662, 0.3295, 0.5327, 0.3571, 0.2337, 0.4303, 0.2565, 0.3773,
                     0.4054, 0.4869, 0.3854, 0.2507, 0.2546, 0.4406, 0.1964, 0.2161, 0.2692, 0.3829,
                     0.4163, 0.2545, 0.4337, 0.4852, 0.5455, 0.5793],
    'pattern-finding': [0.4283, 0.3277, 0.3835, 0.3491, 0.3041, 0.2844, 0.2736, 0.2643, 0.2562, 0.2493,
                        0.3424, 0.3358, 0.3071, 0.4119, 0.3198, 0.3746, 0.4375, 0.6195, 0.3345, 0.4461,
                        0.5409, 0.647, 0.3753, 0.3336, 0.3394, 0.3403, 0.2291, 0.2823, 0.3677, 0.2745,
                        0.3844, 0.3048, 0.397, 0.4018, 0.358, 0.4164],
    'unit-conversion': [0.3966, 0.3627, 0.4418, 0.3658, 0.3313, 0.3093, 0.3025, 0.2963, 0.2913, 0.2868,
                        0.2521, 0.1892, 0.2214, 0.4445, 0.3329, 0.4093, 0.3479, 0.3138, 0.2848, 0.3404,
                        0.3042, 0.3545, 0.4073, 0.3555, 0.3524, 0.2289, 0.2169, 0.2441, 0.3678, 0.2376,
                        0.2879, 0.2472, 0.4043, 0.2693, 0.3096, 0.3229],
    'noskill': [0.4686, 0.4587, 0.4129, 0.4094, 0.4058, 0.3915, 0.379, 0.371, 0.3645, 0.3587,
                0.3008, 0.3303, 0.3505, 0.3767, 0.3549, 0.3845, 0.3424, 0.3101, 0.3317, 0.3246,
                0.3778, 0.4629, 0.443, 0.4016, 0.3848, 0.3321, 0.3267, 0.3434, 0.3791, 0.2957,
                0.3876, 0.4154, 0.4176, 0.4781, 0.5657, 0.5385]
}

# 将数据转化为 DataFrame
df = pd.DataFrame(data)

# 转置数据
df_transposed = df.T

# 保存为 Excel 文件
df_transposed.to_excel('mastery_data_transposed.xlsx', index=True)


