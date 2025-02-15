import os

import numpy as np
import torch
from hgnn_models import HGNN
from torch.nn import Module, Embedding, RNN, Linear, Dropout
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.skill_embedding = Embedding(num_c+1, self.emb_size)
        self.answer_embedding = Embedding(2 + 1, self.emb_size)

        # D Graph
        self.gcn_conv1 = GCNConv(self.emb_size, 8)
        self.gcn_conv2 = GCNConv(8, self.emb_size)
        self.fc_d = Linear(self.emb_size, self.num_c)

        # RNN
        self.rnn_Layer_DG = RNN(self.emb_size*3, self.hidden_size, batch_first=True)
        self.pos = nn.Parameter(torch.rand([500, 500, 1]))
        # output
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        self.sigmoid = nn.Sigmoid()
    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def _generate_edge_index(self, skill):
        """
        根据学生的做题序列 (skill) 生成边索引 (edge_index)，使用题目编号本身作为边索引
        """
        batch_size, seq_len = skill.size()
        all_edge_indices = []

        for b in range(batch_size):
            edges = []
            for i in range(seq_len - 1):
                edges.append([skill[b, i].item(), skill[b, i + 1].item()])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, seq_len-1)
            all_edge_indices.append(edge_index)

        return all_edge_indices  # 返回每个学生的 edge_index 列表
    def forward(self,skill, answer):
        mask = torch.ne(answer, 2).unsqueeze(-1).float()
        skill_embedding = self.skill_embedding(skill)
        answer_embedding = self.answer_embedding(answer)
        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)
        answer = answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)

        # D Graph
        all_edge_indices = self._generate_edge_index(skill)
        all_stu_h = []

        for b in range(skill.shape[0]):
            # 将学生嵌入 stu_h 作为输入进行图卷积
            data = Data(x=self.skill_embedding.weight, edge_index=all_edge_indices[b].to(device))
            b_stu_h = self.gcn_conv1(data.x, data.edge_index)
            b_stu_h = F.relu(b_stu_h)
            b_stu_h = self.gcn_conv2(b_stu_h, data.edge_index)
            all_stu_h.append(b_stu_h.unsqueeze(0))  # 保持 batch 维度

        # 拼接所有学生的嵌入
        all_stu_h = torch.cat(all_stu_h, dim=0)  # [b, num_c+1, emb_size]
        skill_index = skill.unsqueeze(-1) # [b,l,1]
        all_stu_h = torch.gather(all_stu_h, 1, skill_index.expand(-1, -1, self.emb_size)) # (b,l,d)

        # 掩码计算

        all_stu_h = all_stu_h * mask
        # 计算有效长度
        effective_lengths = mask.sum(dim=1).squeeze(-1).long()  # [b] 每个样本的有效长度
        # 根据有效长度选择 self.pos 的对应切片
        expand_pos = torch.stack([self.pos[length - 1] for length in effective_lengths], dim=0)  # [b, 500, 1]
        # 应用 softmax 并使用掩码过滤
        expand_pos = F.softmax(expand_pos * mask, dim=1)
        all_stu_h = torch.sum(all_stu_h * expand_pos, dim=1)  # 在 dim=1 上求加权和，得到 [b, d]
        # 恢复原始形状
        all_stu_h = all_stu_h.unsqueeze(1).expand(-1, skill.shape[1], -1)

        x = skill_answer_embedding

        x_DG = torch.cat((all_stu_h, x), dim=-1) # 学生做题的顺序

        h_DG, _ = self.rnn_Layer_DG(x_DG) # layer的层也许可以改一下？
        h_DG = self.dropout_layer(h_DG)
        h_DG = self.out_layer(h_DG)
        logit_d = self.sigmoid(h_DG)


        # 应该检测一下,y在各个时间步的值
        logit_d, = logit_d[:,:-1,:],

        return self._get_next_pred(logit_d,skill),None