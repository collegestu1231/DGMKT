import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
import torch.nn as nn
from hgnn_models import HGNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
class DKVMN(Module):
    def __init__(self, num_c, emb_size, size_m,G, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.emb_size = emb_size
        self.size_m = size_m
        self.emb_type = emb_type
        self.G = G
        emb = Embedding(G.shape[0], self.emb_size)  # 学生数目=4151
        self.stu = emb(torch.LongTensor([i for i in range(G.shape[0])])).cuda()
        self.gate_fc_k_h = nn.Linear(2 * self.emb_size,1)
        self.gate_fc_v_h = nn.Linear(2 * self.emb_size,1) # 这里有问题,应该再加一个D
        self.gate_fc_k_d = nn.Linear(2 * self.emb_size,1)
        self.gate_fc_v_d = nn.Linear(2 * self.emb_size,1)
        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c + 1, self.emb_size)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.emb_size))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.emb_size))
        self.net = HGNN(in_ch=self.emb_size,
                        n_hid=self.emb_size,
                        n_class=self.emb_size)
        self.gcn_conv1 = GCNConv(self.emb_size, 8)
        self.gcn_conv2 = GCNConv(8, self.emb_size)

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2 + 1, self.emb_size)

        self.fh_layer = Linear(self.emb_size * 2, self.emb_size)
        self.fd_layer = Linear(self.emb_size * 2, self.emb_size)
        self.dropout_layer = Dropout(dropout)
        self.h_layer = Linear(self.emb_size, self.num_c)
        self.d_layer = Linear(self.emb_size, self.num_c)
        self.eh_layer = Linear(self.emb_size, self.emb_size)
        self.ah_layer = Linear(self.emb_size, self.emb_size)
        self.ed_layer = Linear(self.emb_size, self.emb_size)
        self.ad_layer = Linear(self.emb_size, self.emb_size)
        # DGEKT-method
        self.w1 = nn.Linear(self.emb_size, self.emb_size)
        self.w2 = nn.Linear(self.emb_size, self.emb_size)

        self.sigmoid = nn.Sigmoid()
        self.fc_ensemble = nn.Linear(2 * self.emb_size, self.num_c)
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
    def forward(self, student,skill, answer, qtest=False):
        # H Graph
        student = F.one_hot(student - 1, num_classes=self.G.shape[0])
        stu_embedding = self.net(self.stu, self.G)
        stu_h = student.float().matmul(stu_embedding) # b,l,d
        emb_type = self.emb_type
        batch_size = skill.shape[0]
        if emb_type == "qid":
            answer_x = torch.where(answer == 2, torch.tensor([1]).to(device), answer)
            x = skill + self.num_c * answer_x  # b,l
            k = self.k_emb_layer(skill)  # 问题嵌入 # b,l,d
            gate = torch.sigmoid(self.gate_fc_k_h(torch.cat([stu_h,k], dim=-1)))
            k = gate * stu_h + (1 - gate) * k
            v = self.v_emb_layer(x)  # 获得知识增长
            gate = torch.sigmoid(self.gate_fc_v_h(torch.cat([stu_h,v], dim=-1)))
            v = gate * stu_h + (1 - gate) * v
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)
        # Write Process
        e = torch.sigmoid(self.eh_layer(v))
        a = torch.tanh(self.ah_layer(v))
        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1)
        # Read Process
        h = torch.tanh(
            self.fh_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        logit_h = self.h_layer(h)

        # D_Graph
        all_edge_indices = self._generate_edge_index(skill)
        all_stu_h = []
        for b in range(skill.shape[0]):
            # 将学生嵌入 stu_h 作为输入进行图卷积
            data = Data(x=self.k_emb_layer.weight, edge_index=all_edge_indices[b].to(device))
            b_stu_h = self.gcn_conv1(data.x, data.edge_index)
            b_stu_h = F.relu(b_stu_h)
            b_stu_h = self.gcn_conv2(b_stu_h, data.edge_index)
            all_stu_h.append(b_stu_h.unsqueeze(0))  # 保持 batch 维度
        # 拼接所有学生的嵌入
        all_stu_h = torch.cat(all_stu_h, dim=0)  # [b, num_c+1, emb_size]
        skill_index = skill.unsqueeze(-1)  # [b,l,1]
        all_stu_h = torch.gather(all_stu_h, 1, skill_index.expand(-1, -1, self.emb_size))  # (b,l,d)
        mask = torch.ne(answer, 2)
        mask = mask.unsqueeze(-1).float()
        all_stu_h = torch.mean(all_stu_h * mask, dim=1)
        # print(all_stu_h.shape)
        all_stu_h = all_stu_h.unsqueeze(1).expand(-1, skill.shape[1], -1)  # b,l
        batch_size = skill.shape[0]
        if emb_type == "qid":
            answer_x = torch.where(answer == 2, torch.tensor([1]).to(device), answer)
            x = skill + self.num_c * answer_x  # b,l
            k = self.k_emb_layer(skill)  # 问题嵌入 # b,l,d
            gate = torch.sigmoid(self.gate_fc_k_d(torch.cat([all_stu_h, k], dim=-1)))
            k = gate * all_stu_h + (1 - gate) * k
            v = self.v_emb_layer(x)  # 获得知识增长
            gate = torch.sigmoid(self.gate_fc_v_d(torch.cat([all_stu_h, v], dim=-1)))
            v = gate * all_stu_h + (1 - gate) * v
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)
        # Write Process
        e = torch.sigmoid(self.ed_layer(v))
        a = torch.tanh(self.ad_layer(v))
        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1)
        # Read Process
        d = torch.tanh(
            self.fd_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        logit_d = self.d_layer(d)
        theta = self.sigmoid(self.w1(h) + self.w2(d))
        h = theta * h
        d = (1 - theta) * d
        emseble_logit = self.fc_ensemble(torch.cat([h, d], -1))
        logit_h, logit_d, emseble_logit = logit_h[:, :-1, :], logit_d[:, :-1, :], emseble_logit[:, :-1, :]

        return  self._get_next_pred(logit_h, skill), self._get_next_pred(logit_d, skill), self._get_next_pred(emseble_logit,skill),