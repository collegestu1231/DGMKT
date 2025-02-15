import os

import numpy as np
import torch
from hgnn_models import HGNN
from torch.nn import Module, Embedding, RNN, Linear, Dropout
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DKT(Module):
    def __init__(self, num_c, emb_size, G,dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.skill_embedding = Embedding(num_c + 1, self.emb_size)
        self.answer_embedding = Embedding(2 + 1, self.emb_size)
        self.G = G
        emb = Embedding(G.shape[0], self.emb_size)
        self.stu = emb(torch.LongTensor([i for i in range(G.shape[0])])).cuda()
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2+1, self.emb_size)
        self.net = HGNN(in_ch=self.emb_size,
                        n_hid=self.emb_size,
                        n_class=self.emb_size)
        self.change_dim = Linear(self.emb_size * 3, self.emb_size)
        self.rnn_layer = RNN(self.emb_size*3, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    def forward(self,student,skill, answer):
        # print(student)
        student = F.one_hot(student - 1, num_classes=self.G.shape[0])  # 24 500 4151

        stu_embedding = self.net(self.stu, self.G)

        skill_embedding = self.skill_embedding(skill)
        answer_embedding = self.answer_embedding(answer)
        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)
        stu_h = student.float().matmul(stu_embedding)  # [b,l,d]
        answer = answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)
        x = skill_answer_embedding
        x = torch.cat((stu_h, x), dim=-1)
        # x = self.change_dim(x)
        h, _ = self.rnn_layer(x)
        h = self.dropout_layer(h)
        y = self.out_layer(h)

        y = torch.sigmoid(y)  # torch.Size([batch_size,seq_len,num_c])
        y = y[:,:-1,:]

        return self._get_next_pred(y, skill), None