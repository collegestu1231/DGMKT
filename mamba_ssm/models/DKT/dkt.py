import os

import numpy as np
import torch

from torch.nn import Module, Embedding, RNN, Linear, Dropout,LSTM,GRU
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Test_file import plot_radar_chart,find_unique_steps,plot_mastery_heatmap
class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.skill_embedding = Embedding(num_c + 1, self.emb_size)
        self.answer_embedding = Embedding(2 + 1, self.emb_size)
        self.sigmoid = torch.nn.Sigmoid()
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2+1, self.emb_size)
        self.change_dim = Linear(self.emb_size * 2, self.emb_size)
        self.lstm_layer = RNN(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    def forward(self, skill, answer):
        emb_type = self.emb_type
        if emb_type == "qid":
            answer_x = torch.where(answer == 2, torch.tensor([1]).to(device), answer)
            x = skill + self.num_c * answer_x
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")(b,l,d)
        # print('你好',self.lstm_layer)(d,d)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        step, concepts = find_unique_steps(skill[0, :])
        # print(skill[0, :step + 1])
        # print(self.sigmoid(y[0, 0:step + 1, concepts]))
        y = torch.sigmoid(y)  # torch.Size([batch_size,seq_len,num_c])
        y = y[:,:-1,:]
        # 应该检测一下,y在各个时间步的值

        return self._get_next_pred(y, skill), None