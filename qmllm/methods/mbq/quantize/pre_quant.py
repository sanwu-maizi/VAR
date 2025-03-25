import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List

import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.mbq.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq.quantize.auto_scale_wa_var import auto_scale_block_wa
from qmllm.methods.mbq.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq.quantize.auto_scale_var import auto_scale_block, apply_scale
# from qmllm.methods.mbq.quantize.auto_scale_var import auto_scale_var_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix
import logging
import torch.nn.functional as F
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


__all__ = ["run_mbq"]


class GradCacheHook:    
    def __init__(self, vis_masks, cap_masks, log_file="/root/autodl-tmp/quantify/grad_cache.txt", plot_dir="/root/autodl-tmp/quantify/plots"):
        if vis_masks is None or cap_masks is None:
            raise ValueError

        self.hooks = []
        self.steps = {}
        self.grad_dict = {}
        self.grad_matrices = {}

        # 确保日志文件目录存在
        log_dir = os.path.dirname(log_file) or "."  # 如果没有目录，默认当前目录
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 确保绘图目录存在
        self.plot_dir = plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        # 配置日志，保存到本地 .txt 文件
        logging.basicConfig(
            filename=log_file,
            filemode='a',  # 'a' 表示追加模式，'w' 表示覆盖
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("GradCacheHook")

    def cache_grad_hook(self, module, inp, out, name):
        if name not in self.steps:
            self.steps[name] = 0
        if name not in self.grad_dict:
            self.grad_dict[name] = []
        if name not in self.grad_matrices:
            self.grad_matrices[name] = []

        output_grad = out[0]
        # print(name,output_grad is not None)
        if output_grad is not None:
            output_grad = output_grad.float()
            step = self.steps[name]

            # 获取梯度张量的形状 [B, N, C]
            grad_shape = list(output_grad.shape)
            # print("output_grad",output_grad.shape)
            # print("grad_shape",len(grad_shape))
            # B, N, C = grad_shape

            # 计算梯度的平均绝对值
            grad_mean = output_grad.abs().mean()

            # 存储梯度均值
            self.grad_dict[name].append(grad_mean.detach().cpu())

            # 计算批量平均的梯度矩阵 [N, C]
            grad_matrix = output_grad.abs().mean(dim=0)  # [N, C]
            self.grad_matrices[name].append(grad_matrix.detach().cpu())

            # 使用 logger 记录梯度信息到本地 txt 文件
            self.logger.info(f"Step {step} - Layer: {name}")
            self.logger.info(f"  Gradient shape: {grad_shape}")
            self.logger.info(f"  Gradient mean abs value: {grad_mean.item():.6f}")
            self.logger.info(f"  Gradient max abs value: {output_grad.abs().max().item():.6f}")
            self.logger.info(f"  Gradient min abs value: {output_grad.abs().min().item():.6f}")
            self.logger.info("-" * 50)

            self.steps[name] += 1


    def plot_grad_heatmaps(self):
        for name, matrices in self.grad_matrices.items():
            if not matrices:
                continue
          
            # 跳过 ada_lin 层
            if "ada_lin" in name:
                self.logger.info(f"Skipping heatmap for layer: {name}")
                continue

            
            # 取所有步的平均梯度矩阵
            grad_matrix = torch.stack(matrices).mean(dim=0).numpy()  # [N, C]         
            
            # 计算平均绝对梯度（归一化前的）
            avg_abs_grad = np.abs(grad_matrix).mean()

            # 归一化到 [-1, 1]
            grad_max = np.abs(grad_matrix).max()
            if grad_max > 0:  # 避免除以零
                grad_matrix_normalized = grad_matrix / grad_max
            else:
                grad_matrix_normalized = grad_matrix  # 如果最大值是 0，则保持原样
            
            grad_matrix_normalized = grad_matrix_normalized*1e2


            # 绘制热图
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                grad_matrix_normalized,
                cmap='coolwarm',
                vmin=0,
                vmax=1,
                center=0,
                annot=False
            )
            plt.title(f"Gradients of {name}\nAvg Abs Grad: {avg_abs_grad:.2e}")
            plt.xlabel("Token Index")
            plt.ylabel("Channel Index")

            # 添加色条，使用 ax.figure.colorbar
            cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
            cbar.set_label("Gradient Magnitude (Normalized to [-1, 1])")

            # 保存图像
            plot_path = os.path.join(self.plot_dir, f"grad_heatmap_{name.replace('.', '_')}.png")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved gradient heatmap to {plot_path}")
            

    def register_hooks(self, layers):
        for n, m in layers.named_modules():
            if isinstance(m, nn.Linear):
                # print(f"Registering hook for layer.{n}")
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}")
                    )
                )


    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


    def get_grad_dict(self):
        return self.grad_dict
    

    def get_avg_grad_dict(self):
        avg_grad_dict = {}

        for name, grad_values in self.grad_dict.items():
#             mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
#             mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))

#             avg_grad_dict[name] = {
#                 "vis_avg_grad": mean_vis.item(),
#                 "cap_avg_grad": mean_cap.item()
#             }
            # grad_values 是每次前向传播的 grad_mean 列表
            mean_grad = torch.mean(torch.stack(grad_values))
            avg_grad_dict[name] = mean_grad.item()

        return avg_grad_dict
    

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "VAR":
        layers = model.blocks
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if model.__class__.__name__ == "VAR":
        # model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
        print(1)
    else:
        raise NotImplementedError(type(model))


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    patch_nums = inputs.pop("patch_nums", None)
    important_scales = inputs.pop("important_scales", None)
    
    return inputs, vision_mask, caption_mask, patch_nums, important_scales

class NoInplaceWrapper(nn.Module):
    """临时包装器，将inplace操作转换为非inplace操作"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._replace_inplace()
    
    def _replace_inplace(self):
        """替换所有inplace操作"""
        def patch_forward(module):
            old_forward = module.forward
            def new_forward(*args, **kwargs):
                # 替换AdaLNBlock中的inplace操作
                if isinstance(module, nn.Module) and 'AdaLNSelfAttn' in module.__class__.__name__:
                    def no_inplace_forward(self, x, cond_BD, attn_bias=None):
                        x = x.clone()  # 保护输入
                        
                        
                        # device = x.device
                        # self.ln_wo_grad = self.ln_wo_grad.to(device)
                        # self.attn = self.attn.to(device)
                        # self.ffn = self.ffn.to(device)
                        
                        if self.shared_aln:
                            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
                        else:
                            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

                            
                        # print(x.device,scale1.device,shift1.device,x_norm1.device)
                        # 注意力分支
                        x_norm1 = self.ln_wo_grad(x)                    # 归一化
                        x_scaled1 = x_norm1 * (scale1 + 1) + shift1     # 缩放和偏移，非原地
                        attn_out = self.attn(x_scaled1, attn_bias=attn_bias)  # 注意力计算
                        # print(x.device,gamma1.device,attn_out.device)
                        attn_out_scaled = attn_out * gamma1             # 缩放，非原地
                        x = x + self.drop_path(attn_out_scaled)         # 残差连接，非原地

                        # FFN 分支
                        x_norm2 = self.ln_wo_grad(x)                    # 归一化
                        x_scaled2 = x_norm2 * (scale2 + 1) + shift2     # 缩放和偏移，非原地
                        ffn_out = self.ffn(x_scaled2)                   # FFN 计算
                        ffn_out_scaled = ffn_out * gamma2               # 缩放，非原地
                        x = x + self.drop_path(ffn_out_scaled)          # 残差连接，非原地

                        return x
                    
                    return no_inplace_forward(module, *args, **kwargs)
                
                # 替换LayerNormWoGrad中的inplace操作
                elif isinstance(module, nn.Module) and 'FFN' in module.__class__.__name__:
                    def no_inplace_ln_forward(self, x):
                        x = x.clone()  # 保护输入
                        if self.fused_mlp_func is not None:
                            out = self.fused_mlp_func(
                                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                                heuristic=0, process_group=None,
                            )
                            out = self.drop(out)
                        else:
                            out = self.fc1(x)        # 线性变换
                            out = self.act(out)      # 激活函数
                            out = self.fc2(out)      # 线性变换
                            out = self.drop(out)     # dropout
                        return out
                    
                    return no_inplace_ln_forward(module, *args, **kwargs)
                
                # 替换AdaLNSelfAttn中的inplace操作
                elif isinstance(module, nn.Module) and 'SelfAttention' in module.__class__.__name__:
                    def no_inplace_attn_forward(self, x, attn_bias=None):
                        B, L, C = x.shape

                        # 使用 clone() 确保输入不被修改
                        x = x.clone()
                        
                        # device = x.device  # 使用输入 x 的设备

                        # 简单粗暴：强制将所有参数和数据移动到 x.device
                        # self.mat_qkv.weight.data = self.mat_qkv.weight.data.to(device)
                        
                        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
                        main_type = qkv.dtype

                        # qkv: BL3Hc
                        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
                        if using_flash or self.using_xform:
                            q, k, v = qkv.unbind(dim=2)  # q or k or v: BLHc
                            dim_cat = 1
                        else:
                            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # q or k or v: BHLc
                            dim_cat = 2

                        # L2 归一化和缩放，避免原地操作
                        if self.attn_l2_norm:
                            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
                            if using_flash or self.using_xform:
                                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
                            # scale_mul = scale_mul.to(device)
                            q = F.normalize(q, dim=-1) * scale_mul  
                            k = F.normalize(k, dim=-1)  # 不修改 k

                        # 缓存逻辑，避免修改原始 k 和 v
                        if self.caching:
                            if self.cached_k is None:
                                self.cached_k = k.clone()  # 复制 k
                                self.cached_v = v.clone()  # 复制 v
                            else:
                                k_new = torch.cat((self.cached_k, k), dim=dim_cat)
                                v_new = torch.cat((self.cached_v, v), dim=dim_cat)
                                self.cached_k = k_new.clone()  # 更新缓存为新对象
                                self.cached_v = v_new.clone()
                                k = k_new  # 使用新张量
                                v = v_new

                        # Dropout 和注意力计算
                        dropout_p = self.attn_drop if self.training else 0.0
                        if using_flash:
                            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
                        elif self.using_xform:
                            oup = memory_efficient_attention(
                                q.to(dtype=main_type),
                                k.to(dtype=main_type),
                                v.to(dtype=main_type),
                                attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1),
                                p=dropout_p,
                                scale=self.scale
                            ).view(B, L, C)
                        else:
                            # 确保所有输入张量都在正确设备上
                            # query = q.to(device)
                            # key = k.to(device)
                            # value = v.to(device)
                            # attn_mask = attn_bias.to(device) if attn_bias is not None else None
                            
                            oup = slow_attn(
                                query=q,
                                key=k,
                                value=v,
                                scale=self.scale,
                                attn_mask=attn_bias,
                                dropout_p=dropout_p
                            ).transpose(1, 2).reshape(B, L, C)

                        # 投影和 dropout，避免原地操作
                        oup = self.proj(oup)
                        oup = self.proj_drop(oup)

                        return oup
                    
                    return no_inplace_attn_forward(module, *args, **kwargs)
                
                return old_forward(*args, **kwargs)
            return new_forward
        
        # 递归替换所有模块的forward函数
        for module in self.model.modules():
            module.forward = patch_forward(module)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@torch.no_grad()
def run_mbq(
    vae_model,
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    auto_scale=True,
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    distort=False
):
    device = next(model.parameters()).device
    mbq_results = {"scale": []}
    # print(prompt_kwargs)

    layers = get_blocks(model)
    # print(layers)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
#     class Catcher(nn.Module):                              #截获第一层输入层的输入，接着生成第一层的输出，实现深拷贝，为ac量化做准备
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, **kwargs):
#             inps.append(inp)
#             kwargs.append(kwargs)
#             raise ValueError  # early exit to break later inference

#     # patch layer 0 to catch input and kwargs
    # layers[0] = Catcher(layers[0])

    inputs, vision_mask, caption_mask, patch_nums, important_scales = process_input(prompt_inputs, prompt_kwargs)
    
    
    label_B = prompt_kwargs["label_B"]
    x_BLCv_wo_first_l = prompt_kwargs["x_BLCv_wo_first_l"]
    gt_BL = inputs.pop("gt_BL",None)
    use_cache = inputs.pop("use_cache",None)
    
    model=NoInplaceWrapper(model)
    
    # print(model)
    
    # 设置训练模式
    model.train()
    
    # 使用VAR的训练loss
    train_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    
    # 分批处理，最大批次大小为8
    batch_size = 8
    total_samples = label_B.shape[0]  # 假设label_B的第一维是样本数（1000）
    num_batches = (total_samples + batch_size - 1) // batch_size  # 计算总批次数

    # try:
    #     model(**inputs)
    # except ValueError: # work with early exit
    #     pass

    # model.to_cpu()
    # layers[0] = layers[0].module  # restore
    # inps = inps[0]       #截获输入层的最初输入
    # layer_kwargs["use_cache"] = False

    # layers[0] = layers[0].cpu()
    # move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    mbq_results = {
        "scale": [],
    }
    
    # 设置训练模式
    model.train()
    
    # 使用VAR的训练loss
    train_loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    
    
    print("Save gradient...")
    # save gradient
    grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask)        
    grad_cache.register_hooks(layers=layers)

    # 分批计算梯度
    for batch_idx in tqdm.tqdm(range(num_batches), desc="Computing gradients"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        # 提取当前批次的输入
        label_batch = label_B[start_idx:end_idx].to(device)
        x_batch = x_BLCv_wo_first_l[start_idx:end_idx].to(device)
        gt_batch = gt_BL[start_idx:end_idx].to(device) if gt_BL is not None else None

        with torch.enable_grad():
            logits_BLV = model(label_batch, x_batch)
            B, V = label_batch.shape[0], vae_model.vocab_size if hasattr(vae_model, 'vae_local') else 4096
            loss = train_loss(logits_BLV.view(-1, V), gt_batch.view(-1))
            loss = loss.view(B, -1).sum(dim=-1).mean()
            loss.backward()

        # 清理显存
        torch.cuda.empty_cache()

    
    
    # 绘制热图并记录日志
    # grad_cache.plot_grad_heatmaps()
    
    grad_avg_dict = grad_cache.get_avg_grad_dict()
    grad_cache.remove_hooks()
    del grad_cache
    
    # 计算reweight_ratio_dict
    reweight_ratio_dict = {}
    for name, grads in grad_avg_dict.items():
        # important_grad = grads["important_grad"]
        # less_important_grad = grads["less_important_grad"]
        # ratio = important_grad / (important_grad + less_important_grad + 1e-5)
        # reweight_ratio_dict[name] = ratio
        reweight_ratio_dict[name]=None
        
        
    
    # 模拟 VAR 的 forward 逻辑，生成 x_BLC, cond_BD, 和 attn_bias
    B = x_BLCv_wo_first_l.shape[0]
    with torch.cuda.amp.autocast(enabled=False):
        label_B = torch.where(torch.rand(B, device=label_B.device) < model.model.cond_drop_rate,
                              model.model.num_classes, label_B)
        cond_BD = model.model.class_emb(label_B)
        sos = cond_BD
        sos = sos.unsqueeze(1).expand(B, model.model.first_l, -1) + model.model.pos_start.expand(B, model.model.first_l, -1)

        if model.model.prog_si == 0:
            x_BLC = sos
        else:
            x_BLC = torch.cat((sos, model.model.word_embed(x_BLCv_wo_first_l.float())), dim=1)

        ed = model.model.begin_ends[model.model.prog_si][1] if model.model.prog_si >= 0 else model.model.L
        x_BLC += model.model.lvl_embed(model.model.lvl_1L[:, :ed].expand(B, -1)) + model.model.pos_1LC[:, :ed]

    attn_bias = model.model.attn_bias_for_masking[:, :, :ed, :ed]
    cond_BD_or_gss = model.model.shared_ada_lin(cond_BD)

    temp = x_BLC.new_ones(8, 8)
    main_type = torch.matmul(temp, temp).dtype

    inps = x_BLC.to(dtype=main_type)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
    attn_bias = attn_bias.to(dtype=main_type)

    layer_kwargs = {
        "cond_BD": cond_BD_or_gss,
        "attn_bias": attn_bias,
    }
    
    # 创建module_kwargs
    module_kwargs = {
        "model": model,
        "input_feat": x_BLCv_wo_first_l,
    }
    
    # 准备输入
    inps = inps[0].cuda()  # 捕获的输入移到 CPU

    
    torch.cuda.empty_cache()
    
    # solve layer by layer
    # 逐层处理
    for i in tqdm.tqdm(range(len(layers)), desc="Running MBQ..."):
        layer = layers[i]
        layer = layer.to(device)
        named_linears = get_named_linears(layer)

        # 首先，获取所有线性层的输入特征
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach()
            feat_dict[name].append(x.to("cuda"))

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        
        # inps = inps.to(next(layer.parameters()).device)  # 支持多 GPU
        # 获取输出作为下一层的输入
        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(next(layer.parameters()).device)

        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        
        # 合并输入特征
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # 清除 GPU 内存
        torch.cuda.empty_cache()

        # 直接使用传入的 reweight_ratio_dict，无需重新计算
        ans_mask = None  # 移除 reweight 逻辑，设置为 None
        vis_mask = None

        # 执行量化
        if wa_quant:  # wa_quant 默认设为 True
            scales_list = auto_scale_block_wa(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                a_bit=a_bit,
                q_config=q_config,
                input_feat=input_feat,
                ans_mask=None,
                vis_mask=None,
                reweight_ratio_dict=reweight_ratio_dict,
                loss_mode=loss_mode
            )
        else:
           scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                ans_mask=ans_mask,
                vis_mask=vis_mask,
                reweight_ratio_dict=reweight_ratio_dict,
                loss_mode=loss_mode
            )
        
        # 应用缩放
        apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

#         # 量化线性层
#         layer_q = copy.deepcopy(layer)
#         layer_q = layer_q.cuda()
#         named_linears_q = get_named_linears(layer_q)
#         for n, m in named_linears_q.items():
#             new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
#             father_module = get_module_by_name_suffix(layer_q, '.'.join(n.split(".")[:-1]))
#             setattr(father_module, n.split('.')[-1], new_linear)
#             del new_linear, m
#             torch.cuda.empty_cache()
        
#         # 更新输入
#         inps = inps.to(next(layer_q.parameters()).device)
#         inps = layer_q(inps, **layer_kwargs)[0]
#         del layer_q
#         torch.cuda.empty_cache()

        # 将输入移回 CPU
        inps = inps.cuda()
        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].cuda()

        # 将当前层移回 CPU
        layers[i] = layers[i].cuda()

        # 清除 GPU 内存
        torch.cuda.empty_cache()

        # 添加前缀，使名称全局化
        mbq_results["scale"] += append_str_prefix(
            scales_list, get_op_name(model.model, layer) + "."
        )


    return mbq_results


def apply_mbq(model, mbq_results):
    apply_scale(model, mbq_results["scale"])