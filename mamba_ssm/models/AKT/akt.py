import torch
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


import torch
import torch.nn as nn
class AKT(nn.Module):
    def __init__(self, n_question,  d_model, n_blocks, kq_same, dropout, model_type, final_fc_dim=512, n_heads=8,
                 d_ff=1024, l2=1e-5, separate_qa=False):
        super().__init__()  # 初始化父类
        # 初始化参数

        '''
        n_question 110
        n_pid 16891
        d_model 256
        n_blocks 1
        kq_same 1
        dropout 0.05
        model_type akt
       '''
        self.n_question = n_question  # 问题数量
        self.dropout = dropout  # 丢弃率
        #  论文里写了:我们发现使用问题嵌入来映射查询和键更有效。
        self.kq_same = kq_same  # 键和查询是否使用相同的嵌入
        self.n_pid = 0

        self.l2 = l2  # L2正则化系数
        self.model_type = model_type  # 模型类型
        self.separate_qa = separate_qa  # 问题和答案是否使用分开的嵌入
        embed_l = d_model  # 嵌入大小

        # 根据问题难度设置条件嵌入
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)  # 难度嵌入
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)  # 考虑难度的问题嵌入
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)  # 考虑难度的交互嵌入

        # 基本嵌入
        self.q_embed = nn.Embedding(self.n_question + 1, embed_l)  # Question Encoder
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)  # 分开的问题和答案嵌入
        else:
            self.qa_embed = nn.Embedding(2, embed_l)  # Interaction Encoder

        # Knowledge Retriever
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff, kq_same=self.kq_same,
                                  model_type=self.model_type)

        # Prediction layer
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()
        # 初始化参数，特别是难度嵌入
        self.reset()

    def reset(self):
        # 将问题难度的嵌入重置为零
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)


    def forward(self, q_data, qa_data,):

        '''
        q_data torch.Size([24, 200])
        qa_data torch.Size([24, 200])
        q_data 应该真是索引了
        qa_data 应该也是索引了
        '''

        # print(q_data)
        # print(qa_data)
        # 查询问题嵌入
        q_embed_data = self.q_embed(q_data)  # 获取问题嵌入
        # q_embed_data torch.Size([24, 200,256])
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)  # 获取分开的问题和答案嵌入(获取交互嵌入)
        else:
            qa_data = (qa_data - q_data) // self.n_question  # 计算答案类型（正确/错误）

            qa_embed_data = self.qa_embed(qa_data) + q_embed_data  # 将问题嵌入加到答案类型嵌入上

        # 通过模型前向传播

        d_output = self.model(q_embed_data, qa_embed_data)  # 知识检索器

        # 将模型输出与问题嵌入结合
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)  # 隐藏状态h_t和问题嵌入x_t
        output = self.out(concat_q)
        output = output.squeeze(-1)
        output = self.sigmoid(output)

        preds = output[:,1:]  # 输出logits

        return preds, None  # 返回总损失和预测


class Architecture(nn.Module):  # 其实是三层Transformer,第一层处理问题嵌入,后两层处理交互嵌入
    # d_ff  前馈网络的维度
    def __init__(self, n_question, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()  # 初始化父类
        # 初始化模型参数
        self.d_model = d_model  # 注意力层的输入/输出维度
        self.model_type = model_type  # 模型类型标识

        if model_type in {'akt'}:
            # 构建第一组 Transformer 层
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            # 构建第二组 Transformer 层，数量为第一组的两倍
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks * 2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        '''
        q_embed_data: torch.Size([24, 200, 256])
        qa_embed_data: torch.Size([24, 200, 256])
        '''
        # 初始化位置编码
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        # 设置数据流
        y = qa_pos_embed  # 问题和答案的嵌入(交互的嵌入)
        x = q_pos_embed  # 问题嵌入

        # 交互编码
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)  # 应用自注意力，不考虑遮蔽

        # 第二阶段编码，处理问题数据，可能与答案数据交互
        flag_first = True  # 控制标志，用于决定处理模式
        for block in self.blocks_2:
            if flag_first:  # 当前问题处理，不查看当前回答
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # 查看当前回答，不使用位置编码
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        '''
        d_model: 256
        d_feature: 32
        d_ff: 2048
        n_heads: 8
        dropout: 0.05
        kq_same: 1
        '''
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        d_feature * n_heads = d_model
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        '''
        q.shape torch.Size([24, 200, 256])
        而且他不是索引,因为Linear也不收整数。
        '''
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    # 计算query和key的点乘，然后除以d_k的平方根进行缩放
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 维度为[batch_size, num_heads, seq_length, seq_length]
    # q.shape torch.Size([24, 8, 200, 32])
    # 200应该是最大序列长度
    # 从scores张量获取batch_size, num_heads和seq_length的尺寸
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # 生成0到seq_length-1的等差数列张量，并扩展成矩阵，移动到指定设备上
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():  # 不记录梯度
        # 使用mask将scores中的相应位置替换成极小值，以便在softmax时忽略这些位置
        scores_ = scores.masked_fill(mask == 0, -1e32)
        # 对scores应用softmax，归一化处理，就是、gamma_{t,t‘}
        scores_ = F.softmax(scores_, dim=-1)  # 维度仍为[batch_size, num_heads, seq_length, seq_length]
        # 将mask转换为浮点数并移动到设备上，与scores_相乘，再次应用mask
        scores_ = scores_ * mask.float().to(device)
        # 计算scores_的累积和，用于之后的位置效应计算
        distcum_scores = torch.cumsum(scores_, dim=-1)  # 每个点到当前点的累计值
        # 计算scores_在最后一个维度的总和
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # 所有点分数的综合
        # 计算位置差的绝对值，并转换为浮点张量,还有扩展向量的意思捏
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        # 查看维度
        # print('distcum_scores.shape:',distcum_scores.shape)
        # print('disttotal_scores.shape',disttotal_scores.shape)
        # distcum_scores.shape: torch.Size([24, 8, 200, 200])
        # disttotal_scores.shape torch.Size([24, 8, 200, 1])
        # 计算距离分数，考虑到位置效应
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)
        # 对距离分数进行开方处理
        dist_scores = dist_scores.sqrt().detach()  # 不能进行反向传播了捏

    m = nn.Softplus()
    # 使用softplus函数处理gamma，然后加负号并增加维度以适应其他张量
    # print('gamma.shape:',gamma.shape)
    # gamma.shape: torch.Size([8, 1, 1])
    gamma = -1. * m(gamma).unsqueeze(0)
    # a.shape = torch.Size([1, 8, 1, 1])
    # 计算总的位置效应，应用指数函数，并在1e-5到1e5之间进行裁剪
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    # total_effect = torch.clamp(torch.clamp(dist_scores.exp(), min=1e-5), max=1e5)
    # 将计算好的位置效应与原始scores相乘
    scores = scores * total_effect  # s_{t,\tau}

    # 再次应用mask，将指定位置设为极小值
    scores.masked_fill_(mask == 0, -1e32)
    # 对更新后的scores再次应用softmax进行归一化
    scores = F.softmax(scores, dim=-1)  # \alpha_{t,\tau}
    # print('scores.shape:',scores.shape)
    # scores.shape: torch.Size([24, 8, 200, 200])
    # 如果启用zero_pad，向scores的序列长度维度前插入全零矩阵
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    # 应用dropout进行正则化
    scores = dropout(scores)
    # 最后使用scores作为权重，与value进行加权求和，得到输出
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
