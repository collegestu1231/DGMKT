# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

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
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from hgnn_models import HGNN

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


"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model,num_heads,dropout):
        
        super().__init__()

        # below is BinMamba
        self.bin_self_attn = nn.MultiheadAttention(embed_dim=d_model * 2, num_heads=2,
                                                   dropout=0.2)
        self.bin_ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.Dropout(0.2)
        )
        self.bin_norm1 = nn.LayerNorm(d_model * 2)
        self.bin_norm2 = nn.LayerNorm(d_model * 2)

        # below is Mamba
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                               dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.2)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        b, seq_len, _ = x.shape
        mask = future_mask(seq_len)

        if False:
            attn_output, _ = self.bin_self_attn(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2), attn_mask=mask)
            attn_output = attn_output.permute(1, 0, 2)

            x = self.bin_norm1(x + attn_output)
            ffn_output = self.bin_ffn(x)
            x = self.bin_norm2(x + ffn_output)
            return x
        else:
            attn_output, _ = self.self_attn(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2), attn_mask=mask)
            attn_output = attn_output.permute(1, 0, 2)
            x = self.norm1(x + attn_output)
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)
            return x
def future_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
"""


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
    """
    继承了MixerModel,添加了用于NLP任务的线性输出层,使得模型能够根据隐藏状态输出单词的概率分布
    提供从预训练模型加载参数的方法(from_pretrained),以及保存模型到文件的方法(save_pretrained)
    集成了GenerationMixin,这是一个混入类,提供了文本生成相关的方法和功能
    """

    def __init__(
            self,
            config: MambaConfig,  # 模型各种配置参数
            G=None,   # 学生关于问题的关系矩阵
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
        self.change_dim = nn.Linear(d_model * 2, d_model)
        self.skill_embedding = nn.Embedding(num_c + 1, d_model)
        self.answer_embedding = nn.Embedding(2 + 1, d_model)
        self.sig = nn.Sigmoid()
        self.num_c = num_c
        
        # H Graph
        self.G = G
        emb = nn.Embedding(G.shape[0], d_model)  # 学生数目=4151
        self.stu = emb(torch.LongTensor([i for i in range(G.shape[0])])).cuda()
        self.net = HGNN(in_ch=d_model,
                        n_hid=d_model,
                        n_class=d_model)
        
        # basic Model
        self.backbone = MixerModel(
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
        # self.TransformerBlock = TransformerBlock(d_model,2,0.2)
        # output Layer
        self.gate_fc = nn.Linear(2 * d_model, 1)
        self.kt_head = nn.Linear(d_model, num_c, bias=False, **factory_kwargs)  # 构建语言模型输出头,一个线性层,

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def _get_next_pred(self, res, skill):# (B,L-1,D ),
        one_hot = torch.eye(self.num_c, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.num_c).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)

        pred = (res * one_hot_skill).sum(dim=-1)
        return pred



    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, student, skill, answer, difficulty=None, inference_params=None, num_last_tokens=0,
                **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """

        # H Graph
        student = F.one_hot(student-1, num_classes=self.G.shape[0]) # 24 500 4151
        stu_embedding = self.net(self.stu, self.G)
        stu_h = student.float().matmul(stu_embedding) # [b,l,d]
        
        # Basic embedding
        skill_embedding = self.skill_embedding(skill)
        answer_embedding = self.answer_embedding(answer)
        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)
        answer = answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)
        x = skill_answer_embedding # [b,l,2d]
        x = self.change_dim(x) # b,ld


        # concat stu_emb and question emb
        gate = torch.sigmoid(self.gate_fc(torch.cat([stu_h, x], dim=-1)))

        kt_input = gate * stu_h + (1 - gate) * x
        x = self.backbone(kt_input, inference_params=inference_params, **mixer_kwargs)  # 得到隐藏状态了捏

        logits = self.kt_head(x)
        logits = self.sig(logits)
        logits = logits[:, :-1, :]
        return self._get_next_pred(logits, skill), skill_answer_embedding  # 这里加一个skill_answer_embedding

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        #  从预训练的检查点加载模型权重
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)
