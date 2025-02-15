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
        answer_mask = torch.ne(real_answers, -1)  # # 创建一个形状为 [24, 499] 的布尔掩码

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


        # print(y_h)
        loss_kd = self.kd_loss * (torch.sum(torch.abs(y_e - y_d)) + torch.sum(torch.abs(y_e - y_h)))
        ground_truth = ground_truth[:,1:]
        answer_mask = torch.ne(ground_truth, 2)  # # 创建一个形状为 [24, 499] 的布尔掩码
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



