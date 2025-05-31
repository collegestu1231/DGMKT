# Copyright (c) 2023, Albert Gu, Tri Dao.




import math
from functools import partial
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import copy
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin  # 提供常用的生成函数

from hgnn_models import HGNN

from sklearn.decomposition import PCA

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):  # 创建一个Mamba Block,根据配置参数选择不同的归一化层(LayerNorm或RMSNorm)
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    """
    构建Mamba模型的主题类,包含模型的嵌入层,多个处理块,以及一个输出层的规范化。
    通过ModuleLIst管理模型中所有的块,确保它们被有效地迭代处理。
    在前向传播方法中,输入序列首先通过嵌入层,然后一次通过每个块处理,最后应用规范化层

    """

    def __init__(
            self,
            d_model: int,  # 模型维度
            n_layer: int,  # Mamba Block的数目
            d_intermediate: int,  # 时间步
            num_c: int,  # 词汇表维度
            ssm_cfg=None,  # 配置参数
            attn_layer_idx=None,
            attn_cfg=None,
            norm_epsilon: float = 1e-5,  # 配置参数
            rms_norm: bool = False,  # 配置参数,True
            initializer_cfg=None,
            fused_add_norm=False,  # True
            residual_in_fp32=False,  # True
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(num_c, d_model, **factory_kwargs)  # 词嵌入层,将离散的Token转化为连续的向量

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )  # 构建n_layer个Mamba Block，每个Block包括残差连接，LayerNorm,和Mamba层

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )  # 构建了最后一个LayerNorm层(self.norm_f),用于归一模型的最终输出

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )  # 应用一个初始化函数_init_weights,根据层数n_layer调整某些参数的初始值

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        # 定义前向传播流程,词嵌入->n个Block->最后的LayerNorm
        # input_ids.shape torch.Size([batch_size,seq_len])每个ID都是整数

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MambaKTHeadModel(nn.Module, GenerationMixin):
    def __init__(
            self,
            config: MambaConfig,  
            G=None,  
            initializer_cfg=None,
            device=None,
            dtype=None,

    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        num_c = config.num_c
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_num_c_multiple = config.pad_num_c_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if num_c % pad_num_c_multiple != 0:
            num_c += pad_num_c_multiple - (num_c % pad_num_c_multiple)
        self.change_dim_D = nn.Linear(d_model * 3, d_model)
        self.change_dim_H = nn.Linear(d_model*3,d_model)
        self.skill_embedding = nn.Embedding(num_c + 1, d_model)
        self.answer_embedding = nn.Embedding(2 + 1, d_model)
        self.sig = nn.Sigmoid()
        self.num_c = num_c

        self.d_model = d_model
        self.G = G
        self.gcn_conv1 = GCNConv(d_model, 8)
        self.gcn_conv2 = GCNConv(8, d_model)
        emb = nn.Embedding(G.shape[0], d_model)  
        self.stu = emb(torch.LongTensor([i for i in range(G.shape[0])])).cuda()
        self.pos = nn.Parameter(torch.rand([500, 500, 1]))
        self.difficulty = nn.Parameter(torch.FloatTensor(24, 500))
        self.net = HGNN(in_ch=d_model,
                        n_hid=d_model,
                        n_class=d_model)
        self.gate_fc = nn.Linear(2 * d_model, 1)
        self.backbone_H = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            num_c=num_c,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,  # True
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,  # True
            residual_in_fp32=residual_in_fp32,  # True
            **factory_kwargs,
        )  
        self.backbone_D = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            num_c=num_c,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,  # True
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,  # True
            residual_in_fp32=residual_in_fp32,  # True
            **factory_kwargs,
        )  

        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)

        self.fc_d = nn.Linear(d_model, self.num_c)
        self.fc_h = nn.Linear(d_model, self.num_c)
        self.fc_ensemble = nn.Linear(2 * d_model, self.num_c)
        self.lm_head = nn.Linear(d_model, num_c, bias=False, **factory_kwargs) 
        
        self.sigmoid = nn.Sigmoid()
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
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
    def _get_next_pred(self, res, skill):  # (B,L-1,D ),
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, student, skill, answer, cur=None, clo=None, far=None):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        # student b,l

        # H Graph

        temp_answer = answer
        student = F.one_hot(student - 1, num_classes=self.G.shape[0])  # b,l,num_stu
        stu_embedding = self.net(self.stu, self.G)  # torch.Size([stu_num, dim]) 把这里直接做一个UMAP会怎么样呢?
        # print(torch.sum(stu_embedding,dim=-1))
        stu_h = student.float().matmul(stu_embedding)  # [b,l,d]
        # print(stu_h)
        mask = torch.ne(answer, 2).unsqueeze(-1).float()  # [b, l, 1]

        # D Graph
        all_edge_indices = self._generate_edge_index(skill)  # 0~660
        all_stu_h = []
        for b in range(skill.shape[0]): # 对于每个学生进行图卷积
            # 将学生嵌入 stu_h 作为输入进行图卷积
            data = Data(x=self.skill_embedding.weight, edge_index=all_edge_indices[b].to(device))
            b_stu_h = self.gcn_conv1(data.x, data.edge_index)
            b_stu_h = F.relu(b_stu_h)
            b_stu_h = self.gcn_conv2(b_stu_h, data.edge_index)
            all_stu_h.append(b_stu_h.unsqueeze(0))  # 保持 batch 维度
        # 拼接所有学生的嵌入
        all_stu_h = torch.cat(all_stu_h, dim=0)  # [b, num_c+1, emb_size]
        skill_index = skill.unsqueeze(-1)  # [b,l,1]
        all_stu_h = torch.gather(all_stu_h, 1, skill_index.expand(-1, -1, self.d_model))  # (b,l,d)

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
        # print(all_stu_h.shape)

        # Basic embedding
        skill_embedding = self.skill_embedding(skill)

        answer_embedding = self.answer_embedding(answer)
        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)
        answer = answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)
        x = skill_answer_embedding

        x_DG = self.change_dim_D(torch.cat((all_stu_h, x), dim=-1))  # 学生做题的顺序,拼接学生嵌入以及问题嵌入
        x_HG = self.change_dim_H(torch.cat((stu_h, x), dim=-1))  # 学生做了哪些题目

        h_DG = self.backbone_D(x_DG)
        h_HG = self.backbone_H(x_HG)
        logit_h = self.fc_h(h_HG)
        # print(self.num_c)
        # print(logit_h.shape)
        logit_d = self.fc_d(h_DG)
        theta = self.sigmoid(self.w1(h_HG) + self.w2(h_DG))
        h_HG = theta * h_HG
        h_DG = (1 - theta) * h_DG
        emseble_logit = self.fc_ensemble(torch.cat([h_HG, h_DG], -1))

        logit_h, logit_d, emseble_logit = logit_h[:, :-1, :], logit_d[:, :-1, :], emseble_logit[:, :-1, :]
        return self._get_next_pred(logit_h, skill), self._get_next_pred(logit_d, skill), self._get_next_pred(
            emseble_logit, skill)

