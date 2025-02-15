import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_ffw import FeedForward
from hgnn_models import HGNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    # np.triu把下三角全部置为0
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)  # 因为True将会被换成一个巨tm小的值，所以mask最终True位置才是被屏蔽的!
        # mask.shape = torch.Size([1, 1, 200, 200])

    prob_attn = F.softmax(scores, dim=-1)  # 在最里面的[]进行softmax

    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings
    scores = torch.matmul(query, key.transpose(-2, -1))
    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)
    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))
    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):

        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads

        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)  #
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, pos_key_embeds, pos_value_embeds, mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)

        return out


class SAKT(nn.Module):
    # 初始化模型
    def __init__(self, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos=10, drop_prob=0.05):
        """Self-attentive knowledge tracing.
             Arguments:
                 num_items (int): number of items
                 num_skills (int): number of skills
                 embed_size (int): input embedding and attention dot-product dimension
                 num_attn_layers (int): number of attention layers
                 num_heads (int): number of parallel attention heads
                 encode_pos (bool): if True, use relative position embeddings
                 max_pos (int): number of position embeddings to use
                 drop_prob (float): dropout probability
             """

        super(SAKT, self).__init__()
        self.d_model = embed_size
        self.encode_pos = encode_pos
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size, padding_idx=0)
        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)  # 10,40
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.feed_forward = FeedForward(self.d_model, self.d_model, drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()


        # D Graph
        self.gcn_conv1 = GCNConv(self.d_model, 8)
        self.gcn_conv2 = GCNConv(8, self.d_model)
        self.pos = nn.Parameter(torch.rand([500, 500, 1]))

    # 生成模型输入
    def get_inputs(self, skill_inputs, all_stu_h,label_inputs):  # b,l

        skill_inputs = self.skill_embeds(skill_inputs)
        skill_inputs = skill_inputs + all_stu_h
        label_inputs = label_inputs.unsqueeze(-1).float()
        inputs = torch.cat([skill_inputs, skill_inputs], dim=-1)
        inputs[..., :self.d_model] *= label_inputs
        inputs[..., self.d_model:] *= 1 - label_inputs
        return inputs

    # 生成查询向量
    def get_query(self, skill_ids,all_stu_h):  # 问题编码层,也就是E \in \R^{(E×d)}

        skill_ids = self.skill_embeds(skill_ids)
        return skill_ids+all_stu_h

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
    # 模型的前向传播
    def forward(self,skill_ids, skill_inputs, label_inputs,):

        '''
        item_inputs.shape torch.Size([100, 200])
        item_ids.shape torch.Size([100, 200])
        '''


        # D Graph
        all_edge_indices = self._generate_edge_index(skill_ids)
        all_stu_h = []
        for b in range(skill_ids.shape[0]):
            data = Data(x=self.skill_embeds.weight, edge_index=all_edge_indices[b].to(device))
            b_stu_h = self.gcn_conv1(data.x, data.edge_index)
            b_stu_h = F.relu(b_stu_h)
            b_stu_h = self.gcn_conv2(b_stu_h, data.edge_index)
            all_stu_h.append(b_stu_h.unsqueeze(0))
        all_stu_h = torch.cat(all_stu_h, dim=0)  # [b, num_c+1, emb_size]
        skill_index = skill_ids.unsqueeze(-1)  # [b,l,1]
        all_stu_h = torch.gather(all_stu_h, 1, skill_index.expand(-1, -1, self.d_model))  # (b,l,d)
        mask = torch.ne(label_inputs, 2).unsqueeze(-1).float()
        all_stu_h = all_stu_h * mask
        # 计算有效长度
        effective_lengths = mask.sum(dim=1).squeeze(-1).long()  # [b] 每个样本的有效长度
        # 根据有效长度选择 self.pos 的对应切片
        expand_pos = torch.stack([self.pos[length - 1] for length in effective_lengths], dim=0)  # [b, 500, 1]
        # 应用 softmax 并使用掩码过滤
        expand_pos = F.softmax(expand_pos * mask, dim=1)
        all_stu_h = torch.sum(all_stu_h * expand_pos, dim=1)  # 在 dim=1 上求加权和，得到 [b, d]
        # 恢复原始形状
        all_stu_h = all_stu_h.unsqueeze(1).expand(-1, skill_ids.shape[1], -1)
        # sakt
        inputs = self.get_inputs(skill_inputs, all_stu_h,label_inputs)  # [b,l,2d]
        inputs = F.relu(self.lin_in(inputs))  # [b,l,d]
        query = self.get_query(skill_ids,all_stu_h)  # [b,l,d]

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()

        outputs = self.dropout(self.attn_layers[0](query, inputs, inputs, self.encode_pos,
                                                   self.pos_key_embeds, self.pos_value_embeds, mask))

        for l in self.attn_layers[1:]:
            residual = l(query, outputs, outputs, self.encode_pos, self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return self.sigmoid(self.lin_out(outputs).squeeze(-1)[:, 1:]), None
