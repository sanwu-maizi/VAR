import gc
import copy
import torch
import functools
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from models.basic_var import AdaLNSelfAttn

from .qmodule import ScaledActivation, IdentityScaleLayer, ScaledDynamicFC
from collections import defaultdict
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .qmodule import ScaledActivation
from .quantizer import get_module_by_name_suffix
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear

__all__ = ["auto_scale_block_wa_distort"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block_wa_distort(module, module_kwargs, w_bit, a_bit, q_config, input_feat, ans_mask, vis_mask, reweight_ratio_dict, q_input, loss_mode="mae"):

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def _search_module_scale_wa_distort(block, linears2scale: list, layers_name, x, x_q, reweight_ratio, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        x_q = x_q.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x_q)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()

            if isinstance(block, nn.Linear):
                new_block = None
            for fc, fc_name in zip(linears2scale, layers_name):
                # fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                # fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                new_fc = WALinear.from_float(fc, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                
                if isinstance(block, nn.Linear):
                    new_block = copy.deepcopy(new_fc)                   
                else:
                    setattr(block, fc_name, new_fc)
                    
                del new_fc
                torch.cuda.empty_cache()

            x_scale = x_q / (scales.view(1, 1, -1)) 

            if isinstance(block, nn.Linear):
                out = new_block(x_scale, **kwargs)
            else:
                out = block(x_scale, **kwargs)

            if isinstance(out, tuple):
                out = out[0]

            if loss_mode == "mse":
                loss = 0.0
                # if reweight_ratio is not None:  # reweight_ratio 是二维字典，例如 {"attn": {0: 1.5, 1: 1.2, ...}}
                #     for scale_idx, scale_mask in enumerate(scale_masks):
                #         scale_mask_expand = scale_mask.unsqueeze(-1).expand_as(out).to(out.device)
                #         masked_diff = ((org_out - out).float().pow(2) * scale_mask_expand)
                #         scale_loss = masked_diff.sum() / scale_mask_expand.sum()
                #         # 从 reweight_ratio 中获取当前层的对应 scale_idx 的权重
                #         if scale_idx in reweight_ratio:
                #             loss += reweight_ratio[scale_idx] * scale_loss
                #         else:
                #             loss += scale_loss  # 如果没有指定权重，默认权重为 1
                # else:
                #     loss = (org_out - out).float().pow(2).mean().item()
                loss = (org_out - out).float().pow(2).mean().item()
                    # print(loss)
            elif loss_mode == "mae":
                loss = 0.0
                # if reweight_ratio is not None:  # reweight_ratio 是二维字典，例如 {"attn": {0: 1.5, 1: 1.2, ...}}
                #     for scale_idx, scale_mask in enumerate(scale_masks):
                #         scale_mask_expand = scale_mask.unsqueeze(-1).expand_as(out).to(out.device)
                #         masked_diff = ((org_out - out).float().abs() * scale_mask_expand)
                #         scale_loss = masked_diff.sum() / scale_mask_expand.sum()
                #         if scale_idx in reweight_ratio:
                #             loss += reweight_ratio[scale_idx] * scale_loss
                #         else:
                #             loss += scale_loss  # 如果没有指定权重，默认权重为 1
                # else:
                #     loss = (org_out - out).float().abs().mean().item()
                loss = (org_out - out).float().abs().mean().item()
                    # print(loss)

            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales

            # restore the block
            for fc, fc_name in zip(linears2scale, layers_name):
                if isinstance(block, nn.Linear):
                    continue
                else:
                    setattr(block, fc_name, fc)
            
            if isinstance(block, nn.Linear):
                del new_block 
            torch.cuda.empty_cache()
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale_wa_distort(prev_op, layers, layers_name, inp, inp_q, reweight_ratio, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale_wa_distort(module2inspect, layers, layers_name, inp, inp_q, reweight_ratio, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []  # return the searched scales
    
#     def _auto_get_input_feat_distort(inps_q, scales_list=None):
#         # 保存原始状态字典
#         org_sd = {k: v.cpu() for k, v in module.state_dict().items()}

#         # 获取所有 nn.Linear 和 ScaledDynamicFC 模块
#         named_linears = {name: m for name, m in module.named_modules() 
#                          if isinstance(m, (nn.Linear, ScaledDynamicFC))}

#         # 保存被替换的模块
#         modified_modules = {}

#         if scales_list is not None:
#             # 应用缩放（可能替换 nn.Linear 为 ScaledDynamicFC）
#             apply_scale(module, scales_list)
#             module.cuda()

#             # 重新获取 named_linears（结构可能已改变）
#             named_linears = {name: m for name, m in module.named_modules() 
#                              if isinstance(m, (nn.Linear, ScaledDynamicFC))}

#             # 统一处理 ScaledDynamicFC 和 nn.Linear
#             for name, m in named_linears.items():
#                 if isinstance(m, ScaledDynamicFC):
#                     print(f"Retaining ScaledDynamicFC layer: {name}")
#                     # 保存原始 ScaledDynamicFC
#                     modified_modules[name] = copy.deepcopy(m)
#                     m=m.fc
#                     # continue
#                 # 保存原始 nn.Linear
#                 modified_modules[name] = copy.deepcopy(m)
#                 # 创建 WALinear
#                 new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
#                 # 获取父模块并替换
#                 father_module = get_module_by_name_suffix(module, '.'.join(name.split(".")[:-1]))
#                 setattr(father_module, name.split('.')[-1], new_linear)
#                 del new_linear, m
#                 torch.cuda.empty_cache()

#         def cache_input_hook(m, x, y, name, feat_dict):
#             x = x[0]
#             print(f"Hook triggered for layer {name}, input shape: {x.shape}")
#             x = x.detach().cpu()
#             feat_dict[name].append(x)

#         # 注册钩子
#         input_feat_q = defaultdict(list)
#         handles = []
#         for name, module_to_hook in named_linears.items():
#             if isinstance(module_to_hook, ScaledDynamicFC):
#                 print(f"Registering hook to ScaledDynamicFC.fc for {name}")
#                 target_module = module_to_hook.fc
#             else:
#                 target_module = module_to_hook
#             handles.append(
#                 target_module.register_forward_hook(
#                     functools.partial(cache_input_hook, name=name, feat_dict=input_feat_q)
#                 )
#             )

#         # 前向传播
#         inps_q = inps_q.to(next(module.parameters()).device)
#         module(inps_q, **module_kwargs)
#         for h in handles:
#             h.remove()

#         # 合并输入特征
#         input_feat_q = {k: torch.cat(v, dim=0) for k, v in input_feat_q.items()}
#         print(f"_auto_get_input_feat_distort input_feat_q keys: {list(input_feat_q.keys())}")

#         # 恢复被替换的模块
#         for name, original_module in modified_modules.items():
#             father_module = get_module_by_name_suffix(module, '.'.join(name.split(".")[:-1]))
#             setattr(father_module, name.split('.')[-1], original_module)

#         # 恢复原始状态字典
#         module.load_state_dict(org_sd)
#         module.cuda()

#         torch.cuda.empty_cache()

#         return input_feat_q
    
    
    def _auto_get_input_feat_distort(inps_q, scales_list=None):
        # 保存原始状态字典
        org_sd = {k: v.cpu() for k, v in module.state_dict().items()}

        # 获取所有 nn.Linear 模块
        named_linears = {name: m for name, m in module.named_modules() if isinstance(m, (nn.Linear, ScaledDynamicFC))}
            
        # 保存被替换的 nn.Linear 模块
        modified_modules = {}

        if scales_list is not None:
            
            # 应用缩放
            apply_scale(module, scales_list)
            module.cuda()
            
            named_linears = {name: m for name, m in module.named_modules() if isinstance(m, (nn.Linear, ScaledDynamicFC))}
            
            # for name in named_linears:
            #     print(name,named_linears[name])
                
            # print("----------------------")

            # 替换 nn.Linear 为 WALinear
            for name, m in list(named_linears.items()):  # 使用 list 避免修改字典时的迭代问题
                # print(name,m)
                if isinstance(m, ScaledDynamicFC):
                    # print(f"Retaining ScaledDynamicFC layer: {name}")
                    # 保存原始 ScaledDynamicFC
                    modified_modules[name] = copy.deepcopy(m)
                    # print(modified_modules[name])
                    linear_module = m.fc
                    # 创建 WALinear
                    new_linear = WALinear.from_float(linear_module, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                    # print("WALinea替换前", linear_module, m)  # 打印 .fc 和 ScaledDynamicFC
                    # 直接替换 ScaledDynamicFC 的 .fc
                    m.fc = new_linear
                    # print("WALinea替换后", linear_module, m)  # 打印 .fc 和更新后的 ScaledDynamicFC
                    del new_linear, linear_module
                    torch.cuda.empty_cache()
                # 保存原始 nn.Linear
                else:
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name in named_linears and isinstance(named_linears[parent_name], ScaledDynamicFC):
                        # print(f"Skipping nn.Linear submodule {name} of ScaledDynamicFC {parent_name}")
                        continue
                    modified_modules[name] = copy.deepcopy(m)  # 深拷贝原始 nn.Linear 以便恢复
                
                # 创建 WALinear
                    new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                    # print("WALinea替换前",m,module)
                    # 获取父模块并替换
                    father_module = get_module_by_name_suffix(module, '.'.join(name.split(".")[:-1]))
                    setattr(father_module, name.split('.')[-1], new_linear)
                    # print("WALinea替换后",m,module)
                    del new_linear, m
                    torch.cuda.empty_cache()

            # 重新获取 named_linears（现在包含 WALinear）
            named_linears = {name: m for name, m in module.named_modules() if isinstance(m, (WALinear,ScaledDynamicFC))}

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            # print(f"Hook triggered for layer {name}, input shape: {x.shape}")
            x = x.detach().cpu()
            feat_dict[name].append(x)
        
        # print(named_linears)

        input_feat_q = defaultdict(list)
        handles = []
        for name in named_linears:
            module_to_hook = named_linears[name]
            if isinstance(module_to_hook, ScaledDynamicFC):
                # print("ScaledDynamicFChook")
                target_module = module_to_hook.fc
            else:
                target_module = module_to_hook
            handles.append(
                target_module.register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat_q)
                )
            )
            

        inps_q = inps_q.to(next(module.parameters()).device)
        module(inps_q, **module_kwargs)
        for h in handles:
            h.remove()

        input_feat_q = {k: torch.cat(v, dim=0) for k, v in input_feat_q.items()}
        # print(f"_auto_get_input_feat_distort input_feat_q keys: {list(input_feat_q.keys())}")

        # print(modified_modules)
        
        # print("Before",module)
        
        # 恢复被替换的 nn.Linear 模块
        for name, original_linear in modified_modules.items():
            if isinstance(original_linear, ScaledDynamicFC):
                # print(f"Original_linear layer:{name}\n {original_linear}\n{original_linear.fc}")
                set_op_by_name(module, name, original_linear.fc)
                # print("Just alternate",module)
            else :
                # print(f"Original_linear layer:{name}\n {original_linear}\n")
                father_module = get_module_by_name_suffix(module, '.'.join(name.split(".")[:-1]))
                setattr(father_module, name.split('.')[-1], original_linear)
                # print("Just alternate",module)
        
        # print("After",module)
        
        # 恢复原始状态字典
        module.load_state_dict(org_sd)
        module.cuda()

        torch.cuda.empty_cache()

        return input_feat_q
    
#     def _auto_get_input_feat_distort(inps_q, scales_list=None):
#         # 保存原始状态字典
#         org_sd = {k: v.cpu() for k, v in module.state_dict().items()}

#         # 获取所有 nn.Linear 模块
#         named_linears = {name: m for name, m in module.named_modules() if isinstance(m, (nn.Linear, ScaledDynamicFC))}
            
#         # 保存被替换的 nn.Linear 模块
#         modified_modules = {}

#         if scales_list is not None:
            
#             # 应用缩放
#             apply_scale(module, scales_list)
#             module.cuda()

#             # 替换 nn.Linear 为 WALinear
#             for name, m in list(named_linears.items()):  # 使用 list 避免修改字典时的迭代问题
#                 # 保存原始 nn.Linear
#                 modified_modules[name] = copy.deepcopy(m)  # 深拷贝原始 nn.Linear 以便恢复
#                 # 创建 WALinear
#                 new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
#                 # 获取父模块并替换
#                 father_module = get_module_by_name_suffix(module, '.'.join(name.split(".")[:-1]))
#                 setattr(father_module, name.split('.')[-1], new_linear)
#                 del new_linear, m
#                 torch.cuda.empty_cache()

#             # 重新获取 named_linears（现在包含 WALinear）
#             named_linears = {name: m for name, m in module.named_modules() if isinstance(m, WALinear)}

#         def cache_input_hook(m, x, y, name, feat_dict):
#             x = x[0]
#             # print(f"Hook triggered for layer {name}, input shape: {x.shape}")
#             x = x.detach().cpu()
#             feat_dict[name].append(x)

#         input_feat_q = defaultdict(list)
#         handles = []
#         for name in named_linears:
#             handles.append(
#                 named_linears[name].register_forward_hook(
#                     functools.partial(cache_input_hook, name=name, feat_dict=input_feat_q)
#                 )
#             )

#         inps_q = inps_q.to(next(module.parameters()).device)
#         module(inps_q, **module_kwargs)
#         for h in handles:
#             h.remove()

#         input_feat_q = {k: torch.cat(v, dim=0) for k, v in input_feat_q.items()}
#         # print(f"_auto_get_input_feat_distort input_feat_q keys: {list(input_feat_q.keys())}")

#         # 恢复被替换的 nn.Linear 模块
#         for name, original_linear in modified_modules.items():
#             father_module = get_module_by_name_suffix(module, '.'.join(name.split(".")[:-1]))
#             setattr(father_module, name.split('.')[-1], original_linear)

#         # 恢复原始状态字典
#         module.load_state_dict(org_sd)
#         module.cuda()

#         torch.cuda.empty_cache()

#         return input_feat_q
        
        
    if isinstance(module, AdaLNSelfAttn):
        # attn out
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.attn.mat_qkv,
                layers=[module.attn.proj],
                layers_name=["proj"],
                inp=input_feat["attn.proj"],
                inp_q=input_feat_q["attn.proj"],
                reweight_ratio=reweight_ratio_dict.get("attn", None),
            )
        )
        # fc1
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.ln_wo_grad,
                layers=[module.ffn.fc1],
                layers_name=["fc1"],
                inp=input_feat["ffn.fc1"],
                inp_q=input_feat_q["ffn.fc1"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=module.ffn,
            )
        )
        # fc2
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_wa_distort(
                prev_op=module.ffn.act,
                layers=[module.ffn.fc2],
                layers_name=["fc2"],
                inp=input_feat["ffn.fc2"],
                inp_q=input_feat_q["ffn.fc2"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
            )
        )

        
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list

def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        
        # 移动到 GPU
        device = next(module.parameters()).device  # 使用 module 的设备
        prev_op = prev_op.to(device)
        for layer in layers:
            layer = layer.to(device)
        scales = scales.to(device)
        
        # print(prev_op)

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ == "InternLM2RMSNorm" or prev_op.__class__.__name__ == "Qwen2RMSNorm":
            if "fc1" in layer_names[0]:
                new_module = ScaledDynamicFC(layers[0], scales)
                set_op_by_name(module, layer_names[0], new_module)
                # 调整 fc1 的权重，补偿输入端的除法
                new_module.fc.weight.data *= scales.view(1, -1)  # 乘以 scales，作用于输入通道
            else:
                scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            # print(prev_op,scales)
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
            # print(prev_op_name,prev_op)
        elif isinstance(prev_op, ScaledActivation):
            # 如果已经是 ScaledActivation，直接应用缩放
            # print(scales,layers)
            continue
        else:
            # print(prev_op,scales)
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
