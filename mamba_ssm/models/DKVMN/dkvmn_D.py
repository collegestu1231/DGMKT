import os

import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
import torch.nn as nn
from hgnn_models import HGNN
from torch_geometric.data import Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKVMN(Module):
    def __init__(self, num_c, dim_s, size_m,  dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.emb_size = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        self.gate_fc_k = nn.Linear(2 * self.emb_size, 1)
        self.gate_fc_v = nn.Linear(2 * self.emb_size, 1)
        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c + 1, self.emb_size)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.emb_size))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.emb_size))
        # 初始化 self.pos 为 [500, 500, 1]
        self.pos = nn.Parameter(torch.rand([500, 500, 1]))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)
        self.v_emb_layer = Embedding(self.num_c * 2 + 1, self.emb_size)
        self.f_layer = Linear(self.emb_size * 2, self.emb_size)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.emb_size, self.num_c)
        self.e_layer = Linear(self.emb_size, self.emb_size)
        self.a_layer = Linear(self.emb_size, self.emb_size)

        # D Graph
        self.gcn_conv1 = GCNConv(self.emb_size, 8)
        self.gcn_conv2 = GCNConv(8, self.emb_size)

    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def _generate_edge_index(self, skill):
        batch_size, seq_len = skill.size()
        all_edge_indices = []

        for b in range(batch_size):
            edges = []
            for i in range(seq_len - 1):
                edges.append([skill[b, i].item(), skill[b, i + 1].item()])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, seq_len-1)
            all_edge_indices.append(edge_index)

        return all_edge_indices

    def forward(self, skill, answer, qtest=False):

        all_edge_indices = self._generate_edge_index(skill)
        all_stu_h = []
        for b in range(skill.shape[0]):
            data = Data(x=self.k_emb_layer.weight, edge_index=all_edge_indices[b].to(device))
            b_stu_h = self.gcn_conv1(data.x, data.edge_index)
            b_stu_h = F.relu(b_stu_h)
            b_stu_h = self.gcn_conv2(b_stu_h, data.edge_index)
            all_stu_h.append(b_stu_h.unsqueeze(0))
        all_stu_h = torch.cat(all_stu_h, dim=0)  # [b, num_c+1, emb_size]
        skill_index = skill.unsqueeze(-1)
        all_stu_h = torch.gather(all_stu_h, 1, skill_index.expand(-1, -1, self.emb_size))  # (b,l,d)

        # 掩码计算
        mask = torch.ne(answer, 2).unsqueeze(-1).float()  # [b, l, 1]
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

        emb_type = self.emb_type
        batch_size = skill.shape[0]
        if emb_type == "qid":
            answer_x = torch.where(answer == 2, torch.tensor([1]).to(device), answer)
            x = skill + self.num_c * answer_x
            k = self.k_emb_layer(skill)
            gate = torch.sigmoid(self.gate_fc_k(torch.cat([all_stu_h, k], dim=-1)))
            k = gate * all_stu_h + (1 - gate) * k
            v = self.v_emb_layer(x)
            gate = torch.sigmoid(self.gate_fc_v(torch.cat([all_stu_h, v], dim=-1)))
            v = gate * all_stu_h + (1 - gate) * v

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))
        p = torch.sigmoid(p)
        p = p[:, :-1, :]

        return self._get_next_pred(p, skill), None
