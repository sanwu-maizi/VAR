import gc
import torch
import functools
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


from models.basic_var import AdaLNSelfAttn

from .qmodule import ScaledActivation, IdentityScaleLayer, ScaledDynamicFC
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

from collections import defaultdict

__all__ = ["auto_scale_block_distort", "apply_scale"]

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

    ln_has_params = hasattr(ln, "weight") and ln.weight is not None
    if ln_has_params:
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

    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
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
def auto_scale_block_distort(module, module_kwargs, w_bit, q_config, input_feat, ans_mask, vis_mask, reweight_ratio_dict, q_input, loss_mode="mae", scale_masks=None):
    # Weight quantization function
    if w_bit is not None:
        def w_quantize_func(p):
            return pseudo_quantize_tensor(
                p,
                n_bits=w_bit,
                **q_config,
            ).detach()
    else:
        def w_quantize_func(p):
            return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # Find the best scale ratio
    def _search_module_scale_distort(block, linears2scale: list, x, x_q, reweight_ratio=None, kwargs={}):
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
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x_q, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            if loss_mode == "mse":
                loss = 0.0
                if reweight_ratio is not None and scale_masks is not None:
                    for scale_idx, scale_mask in enumerate(scale_masks):
                        scale_mask_expand = scale_mask.unsqueeze(-1).expand_as(out).to(out.device)
                        masked_diff = ((org_out - out).float().pow(2) * scale_mask_expand)
                        scale_loss = masked_diff.sum() / scale_mask_expand.sum()
                        if scale_idx in reweight_ratio:
                            loss += reweight_ratio[scale_idx] * scale_loss
                        else:
                            loss += scale_loss
                else:
                    loss = (org_out - out).float().pow(2).mean().item()
            elif loss_mode == "mae":
                loss = 0.0
                if reweight_ratio is not None and scale_masks is not None:
                    for scale_idx, scale_mask in enumerate(scale_masks):
                        scale_mask_expand = scale_mask.unsqueeze(-1).expand_as(out).to(out.device)
                        masked_diff = ((org_out - out).float().abs() * scale_mask_expand)
                        scale_loss = masked_diff.sum() / scale_mask_expand.sum()
                        if scale_idx in reweight_ratio:
                            loss += reweight_ratio[scale_idx] * scale_loss
                        else:
                            loss += scale_loss
                else:
                    loss = (org_out - out).float().abs().mean().item()

            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)
        if best_ratio == -1:
            print(history)
            raise Exception
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale_distort(prev_op, layers, inp, inp_q, reweight_ratio=None, module2inspect=None, kwargs={}):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale_distort(module2inspect, layers, inp, inp_q, reweight_ratio, kwargs)
        scales = scales.detach().cpu()
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []

    def _auto_get_input_feat_distort(inps_q, scales_list=None):
        # 保存原始状态字典
        org_sd = {k: v.cuda() for k, v in module.state_dict().items()}

        # 获取所有 nn.Linear 模块
        named_linears = {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

        # 记录被替换为 ScaledDynamicFC 的模块
        modified_modules = {}

        if scales_list is not None:
            # 保存替换前的 nn.Linear 模块
            for name, layer in module.named_modules():
                if isinstance(layer, ScaledDynamicFC):
                    modified_modules[name] = layer
                    print("ScaledDynamicFC1")
                    # 恢复原始 nn.Linear
                    set_op_by_name(module, name, layer.fc)

            # 应用缩放，可能会将 nn.Linear 替换为 ScaledDynamicFC
            apply_scale(module, scales_list)
            module.cuda()

            # 重新获取 named_linears，因为模块结构可能已改变
            named_linears = {name: m for name, m in module.named_modules() if isinstance(m, (nn.Linear, ScaledDynamicFC))}
            for name in named_linears:
                if isinstance(named_linears[name], ScaledDynamicFC):
                    print("ScaledDynamicFC2")
                    # 记录新替换的 ScaledDynamicFC
                    modified_modules[name] = named_linears[name]
                    print(modified_modules[name])
                    # 使用内层的 fc 进行量化
                    target_module = named_linears[name].fc
                else:
                    target_module = named_linears[name]
                # 确保量化后的权重在 cuda 上
                target_module.weight.data = w_quantize_func(target_module.weight.data).cuda()
                if hasattr(target_module, 'bias') and target_module.bias is not None:
                    target_module.bias.data = target_module.bias.data.cuda()

            # print(f"After replacement, WALinear modules: {list(named_linears.keys())}")

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cuda()
            feat_dict[name].append(x)
        
        print(named_linears)

        input_feat_q = defaultdict(list)
        handles = []
        for name in named_linears:
            module_to_hook = named_linears[name]
            if isinstance(module_to_hook, ScaledDynamicFC):
                print("ScaledDynamicFChook")
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

        torch.cuda.empty_cache()
        
        print("Before",module)

        # 在加载 org_sd 之前，恢复所有被替换为 ScaledDynamicFC 的模块为原始 nn.Linear
        for name, mod in modified_modules.items():
            print(name,mod,mod.fc)
            set_op_by_name(module, name, mod.fc)
            
        print("After",module)

        # 加载原始状态字典
        module.load_state_dict(org_sd)

        # 恢复状态字典后，将模型移回 cuda
        module.cuda()

        return input_feat_q
     
    if isinstance(module, AdaLNSelfAttn):
        # attn out
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input)
        scales_list.append(
            _auto_get_scale_distort(
                prev_op=module.attn.mat_qkv,
                layers=[module.attn.proj],
                inp=input_feat["attn.proj"],
                inp_q=input_feat_q["attn.proj"],
                reweight_ratio=reweight_ratio_dict.get("attn", None),
            )
        )
        # fc1
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_distort(
                prev_op=module.ln_wo_grad,
                layers=[module.ffn.fc1],
                inp=input_feat["ffn.fc1"],
                inp_q=input_feat_q["ffn.fc1"],
                reweight_ratio=reweight_ratio_dict.get("mlp", None),
                module2inspect=module.ffn,
            )
        )
        # fc2
        input_feat_q = _auto_get_input_feat_distort(inps_q=q_input, scales_list=scales_list)
        scales_list.append(
            _auto_get_scale_distort(
                prev_op=module.ffn.act,
                layers=[module.ffn.fc2],
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