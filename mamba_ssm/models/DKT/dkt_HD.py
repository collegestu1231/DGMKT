import os
# from Test_file import plot_radar_chart,find_unique_steps,plot_mastery_heatmap
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
    def __init__(self, num_c, emb_size, G,dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.skill_embedding = Embedding(num_c+1, self.emb_size)
        self.answer_embedding = Embedding(2 + 1, self.emb_size)
        self.G = G
        emb = Embedding(G.shape[0], self.emb_size)  # 学生数目=4151
        self.stu = emb(torch.LongTensor([i for i in range(G.shape[0])])).cuda()
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2+1, self.emb_size)
        self.net = HGNN(in_ch=self.emb_size,
                        n_hid=self.emb_size,
                        n_class=self.emb_size)
        self.w1 = Linear(self.emb_size, self.emb_size)
        self.w2 = Linear(self.emb_size, self.emb_size)
        self.gcn_conv1 = GCNConv(self.emb_size, 8)
        self.gcn_conv2 = GCNConv(8, self.emb_size)
        self.fc_d = Linear(self.emb_size, self.num_c)
        self.pos = nn.Parameter(torch.rand([500, 500, 1]))
        self.fc_h = Linear(self.emb_size,self.num_c)
        self.rnn_Layer_DG = RNN(self.emb_size*3, self.hidden_size, batch_first=True)
        self.rnn_Layer_HG = RNN(self.emb_size*3, self.hidden_size,batch_first=True)
        self.fc_ensemble = Linear(2 * self.emb_size, self.num_c)
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
    def forward(self,student,skill, answer):
        temp_answer = answer
        mask = torch.ne(answer, 2).unsqueeze(-1).float()
        all_edge_indices = self._generate_edge_index(skill)
        student = F.one_hot(student - 1, num_classes=self.G.shape[0])  # 24 500 4151
        stu_embedding = self.net(self.stu, self.G)
        skill_embedding = self.skill_embedding(skill)
        answer_embedding = self.answer_embedding(answer)
        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)
        stu_h = student.float().matmul(stu_embedding)  # [b,l,d]学生做了哪些题目
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

        # print(all_stu_h.shape) # torch.Size([24, 661, 512])
        answer = answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)
        x = skill_answer_embedding

        x_DG = torch.cat((all_stu_h, x), dim=-1) # 学生做题的顺序
        x_HG = torch.cat((stu_h, x),dim=-1) # 学生做了哪些题目
        # x = self.change_dim(x)

        h_DG, _ = self.rnn_Layer_DG(x_DG) # layer的层也许可以改一下？
        h_HG, _ = self.rnn_Layer_HG(x_HG) # layer的层也许可以改一下？

        logit_h = self.fc_h(h_HG)
        logit_d = self.fc_d(h_DG)
        theta = self.sigmoid(self.w1(h_HG) + self.w2(h_DG))
        h_HG = theta * h_HG
        h_DG = (1 - theta) * h_DG
        emseble_logit = self.fc_ensemble(torch.cat([h_HG, h_DG], -1))
        # 应该检测一下,y在各个时间步的值
        # step, concepts = find_unique_steps(skill[0, :])
        # print(step)
        # print(concepts)
        # print(skill[0, :step + 1])
        # print(temp_answer[0, :step + 1])
        # plot_mastery_heatmap(self.sigmoid(logit_h[0, 0:step + 1, concepts]), concepts)
        logit_h,logit_d,emseble_logit = logit_h[:,:-1,:],logit_d[:,:-1,:],emseble_logit[:,:-1,:]
        # print(self._get_next_pred(logit_h,skill).shape)
        return self._get_next_pred(logit_h,skill), self._get_next_pred(logit_d,skill),self._get_next_pred(emseble_logit,skill),