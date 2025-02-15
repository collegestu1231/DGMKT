# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import json

import os
class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):

        real_answers = real_answers[:, 1:] # real_answers的形状变为[24, 499]
        answer_mask = torch.ne(real_answers, 2)  # # 创建一个形状为 [24, 499] 的布尔掩码

        y_pred = pred_answers[answer_mask].float()
        y_true = real_answers[answer_mask].float()

        loss=nn.BCELoss()(y_pred, y_true)
        return loss, y_pred, y_true

class DUALKTLoss(nn.Module):

    def __init__(self,kd_loss):
        super(DUALKTLoss,self).__init__()
        self.kd_loss = kd_loss
        self.sigmoid = nn.Sigmoid()
        self.lossFun = nn.BCELoss()

    def forward(self,h_logit,d_logit,emseble_logit,ground_truth):
        y_h = self.sigmoid(h_logit)
        y_d = self.sigmoid(d_logit)
        y_e = self.sigmoid(emseble_logit)


        # Kd_loss有问题
        loss_kd = self.kd_loss * (torch.sum(torch.abs(y_e - y_d)) + torch.sum(torch.abs(y_e - y_h)))
        ground_truth = ground_truth[:,1:]
        answer_mask = torch.ne(ground_truth, 2)  # # 创建一个形状为 [24, 499] 的布尔掩码
        #
        y_true = ground_truth[answer_mask].float()
        y_pred = (y_e+y_d+y_h)/3.0
        # print(y_pred)
        y_pred = y_pred[answer_mask].float()
        y_e = y_e[answer_mask].float()
        y_h = y_h[answer_mask].float()
        y_d = y_d[answer_mask].float()
        #print(torch.min(y_pred))
        total_loss = loss_kd+self.lossFun(y_h,y_true)+self.lossFun(y_d,y_true)+self.lossFun(y_e,y_true)
        return total_loss,y_pred,y_true





def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def extract_floats_from_tensor(tensor, json_file_path):
    """
    根据张量中的问题编号，在JSON文件中提取对应的float值。

    参数:
    tensor -- 形状为(batch_size, seq_len)的张量，包含问题编号
    json_file_path -- JSON文件路径，格式为：{"qid": float}

    返回:
    一个新的张量，形状与输入张量相同，包含从JSON文件中提取的float值
    """
    # 读取并解析JSON文件
    with open(json_file_path, 'r') as f:
        qid_to_float = json.load(f)

    # 初始化一个与输入张量形状相同的空张量，用于存储float值
    float_tensor = torch.empty_like(tensor, dtype=torch.float)

    # 遍历输入张量中的每个问题编号
    for i in range(tensor.size(0)):
        for j in range(tensor.size(1)):
            qid = str(tensor[i, j].item())  # 获取当前的问题编号
            # 在JSON文件中查找对应的float值，并赋值给新张量的对应位置

            float_tensor[i, j] = qid_to_float.get(qid, 0.0)  # 如果qid不存在，使用默认值0.0

    return float_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


def get_file_name_identifier(params):
    words = params.model.split('_')
    model_type = words[0]
    if model_type == 'dkt':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2]]
    elif model_type == 'dktplus':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_dm', params.d_model], ['_ts', params.train_set],  ['_h', params.hidden_dim], ['_do', params.dropout], ['_l2', params.l2], ['_r', params.lamda_r], ['_w1', params.lamda_w1], ['_w2', params.lamda_w2]]
    elif model_type == 'dkvmn':
        file_name = [['_b', params.batch_size], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_q', params.q_embed_dim], ['_qa', params.qa_embed_dim], ['_ts', params.train_set], ['_m', params.memory_size], ['_l2', params.l2]]
    elif model_type in {'akt', 'sakt'}:
        file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model], ['_ts', params.train_set], ['_kq', params.kq_same], ['_l2', params.l2]]
    return file_name


def model_isPid_type(model_name):
    words = model_name.split('_')
    is_pid = True if 'pid' in words else False
    return is_pid, words[0]
# import numpy as np
#
# # Original data to be reshaped
# data_new = [
#     [0.2001, 0.4876, 0.4978, 0.4263, 0.4407],
#     [0.2206, 0.4575, 0.3532, 0.3494, 0.4261],
#     [0.3224, 0.3794, 0.3478, 0.4233, 0.3691],
#     [0.3035, 0.3795, 0.3183, 0.3463, 0.3759],
#     [0.2953, 0.3280, 0.2974, 0.3330, 0.3751],
#     [0.2821, 0.2934, 0.2887, 0.3231, 0.3687],
#     [0.2756, 0.2722, 0.2750, 0.3200, 0.3516],
#     [0.2734, 0.2610, 0.2689, 0.3160, 0.3450],
#     [0.2715, 0.2507, 0.2626, 0.3117, 0.3395],
#     [0.2684, 0.2420, 0.2563, 0.3074, 0.3340],
#     [0.5609, 0.4826, 0.2909, 0.2335, 0.2651],
#     [0.5801, 0.5227, 0.2678, 0.1758, 0.3089],
#     [0.5244, 0.5430, 0.2860, 0.2103, 0.3345],
#     [0.3597, 0.3392, 0.3624, 0.3964, 0.3281],
#     [0.6295, 0.5375, 0.3208, 0.3095, 0.3143],
#     [0.3640, 0.5101, 0.4065, 0.3796, 0.3686],
#     [0.2034, 0.3797, 0.3992, 0.3004, 0.3362],
#     [0.3863, 0.3622, 0.6075, 0.3766, 0.2906],
#     [0.2805, 0.3415, 0.3167, 0.2512, 0.3155],
#     [0.4073, 0.3845, 0.4534, 0.3594, 0.3089],
#     [0.3592, 0.4305, 0.5062, 0.2868, 0.3503],
#     [0.4028, 0.4704, 0.5932, 0.3252, 0.4345],
#     [0.3209, 0.4095, 0.3715, 0.4100, 0.4059],
#     [0.2387, 0.3114, 0.3166, 0.3479, 0.3668],
#     [0.2718, 0.3462, 0.3241, 0.3857, 0.3504],
#     [0.2756, 0.5166, 0.3141, 0.2202, 0.3131],
#     [0.2312, 0.2516, 0.2154, 0.2006, 0.3105],
#     [0.2255, 0.2263, 0.2672, 0.2373, 0.3114],
#     [0.3025, 0.3964, 0.3559, 0.3873, 0.3625],
#     [0.2564, 0.4843, 0.2484, 0.2186, 0.2823],
#     [0.3668, 0.3335, 0.3808, 0.3973, 0.3161],
#     [0.3357, 0.2994, 0.3028, 0.2454, 0.3630],
#     [0.4528, 0.4653, 0.4444, 0.4672, 0.3769],
#     [0.3989, 0.5471, 0.3910, 0.3002, 0.4390],
#     [0.3732, 0.6059, 0.3662, 0.2709, 0.5323],
#     [0.3769, 0.6255, 0.4047, 0.2899, 0.5138],
# ]
#
# # Converting to numpy array and reshaping to (5, 36)
# reshaped_data_new = np.array(data_new).reshape(5, 36)
# reshaped_data_new
#
#
#
