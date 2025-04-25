import os
import torch
import random
import numpy as np
from models import build_vae_var
from qmllm.methods.mbq.quantize.pre_quant import run_mbq, apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act
import argparse
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms
from test import write_save_folder_to_file
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# 原有的辅助函数保持不变
def get_scale_importance_masks(patch_nums, important_scales):
    total_tokens = sum(pn ** 2 for pn in patch_nums)
    important_mask = torch.zeros(total_tokens)
    less_important_mask = torch.zeros(total_tokens)
    cur_pos = 0
    for i, pn in enumerate(patch_nums):
        num_tokens = pn ** 2
        if i < important_scales:
            important_mask[cur_pos:cur_pos + num_tokens] = 1.0
        else:
            less_important_mask[cur_pos:cur_pos + num_tokens] = 1.0
        cur_pos += num_tokens
    return important_mask, less_important_mask

def get_scale_masks(patch_nums):
    total_tokens = sum(pn ** 2 for pn in patch_nums)
    scale_masks = []
    cur_pos = 0
    for pn in patch_nums:
        num_tokens = pn ** 2
        mask = torch.zeros(total_tokens)
        mask[cur_pos:cur_pos + num_tokens] = 1.0
        scale_masks.append(mask)
        cur_pos += num_tokens
    return scale_masks

def compute_mse(original, quantized):
    original = original.to('cuda:0')
    quantized = quantized.to('cuda:0')
    mse = torch.mean((original - quantized) ** 2)
    return mse.item()

def var_mbq_entry(
    vae_model,
    model,
    inputs,
    prompt_inputs,
    run_mbq_process: bool,
    pseudo_quant: bool,
    scale_path: str = None,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 16,
    a_bit: int = 16,
    wa_quant: bool = False,
    important_scales: int = 9,
    reweight: bool = True,
    distort: bool = False,
    loss_mode: str = "mae",
    patch_nums: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
):
    q_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
    }
    assert scale_path is not None
    
    if run_mbq_process:
        mbq_results = run_mbq(
            vae_model=vae_model,
            model=model,
            prompt_inputs=prompt_inputs,
            prompt_kwargs=inputs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
        )
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        torch.save(mbq_results, scale_path)
        print(f"MBQ results saved at {scale_path}")
        
    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cuda")
        apply_mbq(model, mbq_results)
        if not wa_quant:
            pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model, w_bit=w_bit, a_bit=a_bit)
    
    return model

# 修改后的 LayerOutputHook：取平均值
class LayerOutputHook:
    def __init__(self):
        self.outputs = {}

    def hook_fn(self, module, input, output, layer_name):
        # output 是 [B, ...] 的张量，取所有样本的平均值
        mean_output = output.detach().mean(dim=0).cpu()  # 沿着 batch 维度取平均
        self.outputs[layer_name] = mean_output

    def register_hooks(self, model):
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self.hook_fn(m, i, o, n)
                )
                hooks.append(hook)
        return hooks

    def remove_hooks(self, hooks):
        for hook in hooks:
            hook.remove()

# 修改后的 plot_layer_differences：处理平均值后的输出
def plot_layer_differences(original_outputs, quantized_outputs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for layer_name in original_outputs:
        if layer_name not in quantized_outputs:
            continue
        
        orig_output = original_outputs[layer_name].flatten()
        quant_output = quantized_outputs[layer_name].flatten()
        
        mse = compute_mse(orig_output, quant_output)
        
        plt.figure(figsize=(10, 6))
        plt.plot(orig_output.numpy()[:1000], label="Original", alpha=0.7)
        plt.plot(quant_output.numpy()[:1000], label="Quantized", alpha=0.7)
        plt.title(f"Layer: {layer_name}\nMSE: {mse:.6f}")
        plt.xlabel("Output Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(save_dir, f"{layer_name.replace('.', '_')}_diff.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved difference plot for {layer_name} at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='/root/autodl-tmp/pretrained_models/var_d30.pth')
    parser.add_argument("--vae_ckpt", type=str, default='/root/autodl-tmp/pretrained_models/vae_ch160v4096z32.pth')
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--scale_path", type=str, default="/root/autodl-tmp/quantify/var_mbq_scales_16.pt")
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--important_scales", type=int, default=3)
    parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/outputs")
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--num_iter', type=int, default=64)
    parser.add_argument('--wa_quant', type=bool, default=False)
    args = parser.parse_args()

    print(f"Quantify-256-w_bit{args.w_bit}-a_bit{args.a_bit}-image{args.num_images}-wa_quant {args.wa_quant}")

    # 设置随机种子
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    output_dir = args.output_dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化VAR模型
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=False,
    )

    # 加载checkpoints
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cuda'), strict=True)
    var.load_state_dict(torch.load(args.ckpt_path, map_location='cuda'), strict=True)
    vae.eval()
    var.train()

    # 准备示例输入：10个样本
    B = 100  # 减少到 10 个样本
    batch_size = B  # 一次性处理所有样本
    def normalize_01_into_pm1(x):
        return x.add(x).add_(-1)
    
    val_aug = transforms.Compose([
        transforms.Resize(round(1.125 * 256), interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    
    parent_folder = "/root/autodl-tmp/val"
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    subfolders.sort()
    if len(subfolders) < B:
        raise ValueError(f"子文件夹数量 ({len(subfolders)}) 小于批量大小 B ({B})")
    subfolders = subfolders[:B]  # 只取前 10 个子文件夹

    inp_B3HW = []
    labels = list(range(B))  # 标签为 0 到 9
    for i in range(B):
        folder = os.path.join(parent_folder, subfolders[i])
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print(f"警告: 文件夹 '{folder}' 中没有图片，跳过")
            continue
        selected_file = random.choice(image_files)
        image_path = os.path.join(folder, selected_file)
        image = Image.open(image_path).convert('RGB')
        image = val_aug(image)
        inp_B3HW.append(image)

    num_batches = 1  # 只有 1 个批次
    gt_idx_Bl_all = []
    x_BLCv_wo_first_l_all = []
    device = torch.device("cuda")

    print("Processing images through VAE...")
    for batch_idx in tqdm(range(num_batches), desc="VAE Encoding"):
        inp_batch = torch.stack(inp_B3HW).to(device)
        gt_idx_Bl = vae.img_to_idxBl(inp_batch)
        x_batch = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
        gt_idx_Bl_all.append(torch.cat(gt_idx_Bl, dim=1))
        x_BLCv_wo_first_l_all.append(x_batch.cpu())
        del inp_batch, gt_idx_Bl, x_batch
        torch.cuda.empty_cache()

    gt_BL = torch.cat(gt_idx_Bl_all, dim=0).to(device)
    x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l_all, dim=0).to(device)

    # 输入数据
    prompt_kwargs = {
        "label_B": torch.tensor(labels, device=device),
        "x_BLCv_wo_first_l": x_BLCv_wo_first_l,
        "gt_BL": gt_BL,
    }
    
    important_mask, less_important_mask = get_scale_importance_masks(patch_nums, args.important_scales)
    scale_masks = get_scale_masks(patch_nums)
    prompt_inputs = {
        "vision_mask": important_mask,
        "caption_mask": less_important_mask,
        "scale_masks": scale_masks,
        "patch_nums": patch_nums,
        "important_scales": args.important_scales
    }

    # 第1步：原始模型推理并记录输出（取平均值）
    print("Step 1: Running inference with original model...")
    original_hook = LayerOutputHook()
    original_hooks = original_hook.register_hooks(var)
    
    with torch.no_grad():
        original_output = var(label_B=prompt_kwargs["label_B"], x_BLCv_wo_first_l=prompt_kwargs["x_BLCv_wo_first_l"])
    
    original_layer_outputs = original_hook.outputs
    original_hook.remove_hooks(original_hooks)
    del original_hooks, original_hook

    # 释放原始模型
    print("Releasing original model...")
    del vae, var
    torch.cuda.empty_cache()

    # 第2步：加载并量化新模型
    print("Step 2: Loading and quantizing new model...")
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cuda'), strict=True)
    var.load_state_dict(torch.load(args.ckpt_path, map_location='cuda'), strict=True)
    vae.eval()
    var.train()

    quantized_var = var_mbq_entry(
        vae_model=vae,
        model=var,
        inputs=prompt_kwargs,
        prompt_inputs=prompt_inputs,
        run_mbq_process=False,
        pseudo_quant=True,
        scale_path=args.scale_path,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        wa_quant=args.wa_quant,
        important_scales=args.important_scales,
        reweight=True,
        patch_nums=patch_nums,
        distort=True
    )
    quantized_var.eval()

    # 第3步：量化模型推理并记录输出（取平均值）
    print("Step 3: Running inference with quantized model...")
    quantized_hook = LayerOutputHook()
    quantized_hooks = quantized_hook.register_hooks(quantized_var)
    
    with torch.no_grad():
        quantized_output = quantized_var(label_B=prompt_kwargs["label_B"], x_BLCv_wo_first_l=prompt_kwargs["x_BLCv_wo_first_l"])
    
    quantized_layer_outputs = quantized_hook.outputs
    quantized_hook.remove_hooks(quantized_hooks)
    del quantized_hooks, quantized_hook

    # 第4步：对比并可视化每层输出差异
    print("Step 4: Comparing and plotting layer output differences...")
    plot_dir = os.path.join(args.output_dir, f"layer_diff_w{args.w_bit}_a{args.a_bit}_wa{int(args.wa_quant)}")
    print(original_layer_outputs)
    print("--------")
    print(quantized_layer_outputs)
    plot_layer_differences(original_layer_outputs, quantized_layer_outputs, plot_dir)

    # 第5步：生成图像
    print("Step 5: Generating images with quantized model...")
    num_images = args.num_images
    num_iter = args.num_iter
    batch_size = num_iter
    cfg = args.cfg
    save_folder = os.path.join(output_dir, f"Quantify-256-w_bit{args.w_bit}-a_bit{args.a_bit}-image{num_images}-wa_quant{args.wa_quant}")
    os.makedirs(save_folder, exist_ok=True)
    num_steps = num_images // batch_size + 1

    class_name = 1000
    assert num_images % class_name == 0
    class_label_gen_world = np.arange(0, class_name).repeat(num_images // class_name)
    
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    used_time = 0.0
    gen_img_cnt = 0
    
    for i in range(num_steps):
        print(f"Generation step {i}/{num_steps}")
        labels_gen = class_label_gen_world[batch_size * i:min(batch_size * (i + 1), num_images)]
        label_B = torch.tensor(labels_gen, device=device)
        B = len(labels_gen)

        torch.cuda.synchronize()
        event_start.record()

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                recon_B3HW = quantized_var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed, more_smooth=False)

        event_end.record()
        torch.cuda.synchronize()

        if i >= 1:
            elapsed_time = event_start.elapsed_time(event_end) / 1000.0
            used_time += elapsed_time
            gen_img_cnt += batch_size
            print(f"Generating {gen_img_cnt} images takes {used_time:.5f} seconds, {used_time / gen_img_cnt:.5f} sec per image")

        transformed_tensor = recon_B3HW.clamp(0, 1)
        transformed_tensor = (transformed_tensor * 255).round().byte()
        image_arrays = [transformed_tensor[b_id].permute(1, 2, 0).cpu().numpy() for b_id in range(transformed_tensor.size(0))]
        images = [Image.fromarray(img_array) for img_array in image_arrays]

        for b_id, img in enumerate(images):
            img_id = i * batch_size + b_id
            file_name = f"{str(img_id).zfill(5)}.png"
            img.save(os.path.join(save_folder, file_name))

    output_txt_path = os.path.join("/root/autodl-tmp/outputs", "results.txt")
    os.makedirs("/root/autodl-tmp/outputs", exist_ok=True)
    sec_per_image = used_time / gen_img_cnt
    write_save_folder_to_file(output_txt_path, save_folder, num_images, sec_per_image)

    torch.cuda.empty_cache()