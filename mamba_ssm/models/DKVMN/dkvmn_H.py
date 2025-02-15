import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
import torch.nn as nn
from hgnn_models import HGNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.gate_fc_k = nn.Linear(2 * self.emb_size, 1)
        self.gate_fc_v = nn.Linear(2 * self.emb_size,1)
        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c + 1, self.emb_size)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.emb_size))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.emb_size))
        self.net = HGNN(in_ch=self.emb_size,
                        n_hid=self.emb_size,
                        n_class=self.emb_size)
        self.change_dim_k = Linear(self.emb_size * 2, self.emb_size)
        self.change_dim_v = Linear(self.emb_size * 2, self.emb_size)
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2 + 1, self.emb_size)

        self.f_layer = Linear(self.emb_size * 2, self.emb_size)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.emb_size, self.num_c)

        self.e_layer = Linear(self.emb_size, self.emb_size)
        self.a_layer = Linear(self.emb_size, self.emb_size)

    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def forward(self, student,skill, answer, qtest=False):
        # H Graph
        student = F.one_hot(student - 1, num_classes=self.G.shape[0])
        stu_embedding = self.net(self.stu, self.G)
        stu_h = student.float().matmul(stu_embedding) # b,l,d

        emb_type = self.emb_type
        batch_size = skill.shape[0]
        if emb_type == "qid":
            answer_x = torch.where(answer == 2, torch.tensor([1]).to(device), answer)
            # print(answer_x)
            x = skill + self.num_c * answer_x  # b,l
            k = self.k_emb_layer(skill)  # 问题嵌入 # b,l,d
            gate = torch.sigmoid(self.gate_fc_k(torch.cat([stu_h,k], dim=-1)))

            k = gate * stu_h + (1 - gate) * k

            v = self.v_emb_layer(x)  # 获得知识增长
            gate = torch.sigmoid(self.gate_fc_v(torch.cat([stu_h, v], dim=-1)))
            v = gate * stu_h + (1 - gate) * v

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
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
        # print(f"p: {p.shape}") b,l,d
        p = p[:, :-1, :]

        return self._get_next_pred(p, skill), None