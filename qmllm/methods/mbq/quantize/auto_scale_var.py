import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from models.basic_var import AdaLNSelfAttn

from .qmodule import ScaledActivation
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):   #计算权重的缩放因子
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):            #计算激活值的缩放因子
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    # scales = scales.to(ln.weight.device)

    # 检查 ln 是否有参数（elementwise_affine=False 时没有参数）
    ln_has_params = hasattr(ln, "weight") and ln.weight is not None
    if ln_has_params:  # 如果 ln 有参数
        ln.weight.div_(scales)  
        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias.div_(scales)
            
#     ln.weight.div_(scales)
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

    # scales = scales.to(fc1.weight.device)

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
def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat, ans_mask, vis_mask, reweight_ratio_dict, loss_mode="mae"):

    # firstly, get the weight quantize function
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

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, reweight_ratio=None, kwargs={}):
        # w: co, ci
        # x: n, ci
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

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
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            # loss = (
            #     (org_out - out).float().pow(2).mean().item()
            # )  # float prevents overflow

            if loss_mode == "mse":
                if ans_mask is not None and vis_mask is not None:
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda()
                    masked_diff_ans = ((org_out - out).float().pow(2) * ans_mask_expand)
                    masked_diff_vis = ((org_out - out).float().pow(2) * vis_mask_expand)
                    if reweight_ratio is not None:
                        loss = masked_diff_ans.sum() / ans_mask_expand.sum() + reweight_ratio * (masked_diff_vis.sum() / vis_mask_expand.sum())
                    else:
                        loss = (
                            (org_out - out).float().pow(2).mean().item()
                        ) 
                elif ans_mask is not None and vis_mask is None:
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    masked_diff = ((org_out - out).float().pow(2) * ans_mask_expand)
                    loss = masked_diff.sum() / ans_mask_expand.sum() 
                else:
                    loss = (
                        (org_out - out).float().pow(2).mean().item()
                    )  # float prevents overflow
            elif loss_mode == "mae":
                if ans_mask is not None and vis_mask is not None:
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda()
                    masked_diff_ans = ((org_out - out).float().abs() * ans_mask_expand)
                    masked_diff_vis = ((org_out - out).float().abs() * vis_mask_expand)
                    if reweight_ratio is not None:
                        loss = (masked_diff_ans.sum() + reweight_ratio * masked_diff_vis.sum()) / (ans_mask_expand.sum() + vis_mask_expand.sum())
                    else:
                        loss = (
                            (org_out - out).float().abs().mean().item()
                        ) 
                elif ans_mask is not None and vis_mask is None:
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out)
                    masked_diff = ((org_out - out).float().abs() * ans_mask_expand)
                    loss = masked_diff.sum() / ans_mask_expand.sum() 
                else:
                    loss = (
                        (org_out - out).float().abs().mean().item()
                    )  # float prevents overflow

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
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, reweight_ratio=None, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, reweight_ratio, kwargs)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []  # return the searched scales
    
    if isinstance(module, AdaLNSelfAttn):  # 假设 AdaLNSelfAttn 是你的自定义类
        # attention input: quantize mat_qkv
        # scales_list.append(
        #     _auto_get_scale_wa(
        #         prev_op=module.ln_wo_grad,  # 假设 ln_wo_grad 是前置归一化层
        #         layers=[module.attn.mat_qkv],
        #         layers_name=["mat_qkv"],
        #         inp=input_feat["attn.mat_qkv"],
        #         reweight_ratio=reweight_ratio_dict.get("attn", None),
        #         module2inspect=module.attn,
        #         kwargs=module_kwargs,
        #     )
        # )

        # attn out: quantize proj
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.mat_qkv,
                layers=[module.attn.proj],
                inp=input_feat["attn.proj"],
                reweight_ratio=reweight_ratio_dict.get("attn", None),
            )
        )

        # fc1: quantize fc1
#         scales_list.append(
#             _auto_get_scale(
#                 prev_op=module.ln_wo_grad,  # 再次使用 ln_wo_grad 作为 MLP 的前置归一化
#                 layers=[module.ffn.fc1],
#                 inp=input_feat["ffn.fc1"],
#                 reweight_ratio=reweight_ratio_dict.get("mlp", None),
#                 module2inspect=module.ffn,
#             )
#         )

#         # fc2: quantize fc2
#         scales_list.append(
#             _auto_get_scale(
#                 prev_op=module.ffn.fc1,
#                 layers=[module.ffn.fc2],
#                 inp=input_feat["ffn.fc2"],
#                 reweight_ratio=reweight_ratio_dict.get("mlp", None),
#             )
#         )

        # # ada_lin: quantize ada_lin.1 (可选)
        # scales_list.append(
        #     _auto_get_scale(
        #         prev_op=module.ada_lin[0],  # 假设 SiLU 是前置激活
        #         layers=[module.ada_lin[1]],
        #         inp=input_feat["ada_lin.1"],
        #         reweight_ratio=reweight_ratio_dict.get("ada", None),
        #     )
        # )
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

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ == "InternLM2RMSNorm" or prev_op.__class__.__name__ == "Qwen2RMSNorm":
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        # elif isinstance(prev_op, AdaLNSelfAttn):  # 针对VAR模型的特定层
            # 处理AdaLNSelfAttn的缩放
            # scale_ln_fcs(prev_op.ln_wo_grad, layers, scales)  # 使用ln_wo_grad进行缩放
        else:
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
