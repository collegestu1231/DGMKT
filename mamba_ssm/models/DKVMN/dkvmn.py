import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DKVMN(Module):
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c+1, self.dim_s)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))


        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2+1, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, self.num_c)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def forward(self, skill, answer, qtest=False):


        emb_type = self.emb_type
        batch_size = skill.shape[0]
        if emb_type == "qid":
            answer_x = torch.where(answer == 2, torch.tensor([1]).to(device), answer)
            # print(answer_x)
            x = skill + self.num_c * answer_x # b,l

            k = self.k_emb_layer(skill) # 问题嵌入

            v = self.v_emb_layer(x) # 获得知识增长
            # print(v.shape)


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
        # print(f"p: {p.shape}")
        p = p[:, :-1, :]
        return self._get_next_pred(p, skill), None