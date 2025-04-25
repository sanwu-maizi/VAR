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

def get_scale_importance_masks(patch_nums, important_scales):
    """生成基于scale的重要性mask"""
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
    """
    为每个scale生成独立的mask。
    
    Args:
        patch_nums (tuple): 包含每个scale的patch数量，例如 (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    Returns:
        list of torch.Tensor: 每个scale的mask列表，长度等于len(patch_nums)
    """
    # 计算总token数
    total_tokens = sum(pn ** 2 for pn in patch_nums)
    
    # 初始化mask列表
    scale_masks = []
    
    # 当前token位置
    cur_pos = 0
    
    # 为每个scale生成mask
    for pn in patch_nums:
        num_tokens = pn ** 2  # 当前scale的token数量
        # 创建全0的mask
        mask = torch.zeros(total_tokens)
        # 将当前scale的token设为1
        mask[cur_pos:cur_pos + num_tokens] = 1.0
        # 添加到mask列表
        scale_masks.append(mask)
        # 更新位置
        cur_pos += num_tokens
    
    return scale_masks

def enable_grad_for_model(model):
    """启用模型的梯度计算"""
    for param in model.parameters():
        param.requires_grad = True
    model.train()  # 设置为训练模式以启用梯度计算
    return model

def disable_grad_for_model(model):
    """禁用模型的梯度计算"""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # 恢复为评估模式
    return model

def compute_mse(original, quantized):
    """计算两个张量之间的均方误差"""
    original = original.to('cuda:0')
    quantized = quantized.to('cuda:0')
    mse = torch.mean((original - quantized) ** 2)
    return mse.item()

def compute_mae(original, quantized):
    """计算两个张量之间的平均绝对误差"""
    original = original.to('cuda:0')
    quantized = quantized.to('cuda:0')
    mae = torch.mean(torch.abs(original - quantized))
    return mae.item()

def compute_max_error(original, quantized):
    """计算两个张量之间的最大绝对误差"""
    original = original.to('cuda:0')
    quantized = quantized.to('cuda:0')
    max_error = torch.max(torch.abs(original - quantized))
    return max_error.item()

def compute_relative_error(original, quantized, eps=1e-8):
    """计算两个张量之间的平均相对误差"""
    original = original.to('cuda:0')
    quantized = quantized.to('cuda:0')
    relative_error = torch.mean(torch.abs(original - quantized) / (torch.abs(original) + eps))
    return relative_error.item()

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
    
    # 保存原始模型的权重（深拷贝以避免修改）
    

    if run_mbq_process:
        # 将patch_nums添加到prompt_inputs中
        # prompt_inputs["patch_nums"] = patch_nums
        # prompt_inputs["important_scales"] = important_scales
        
        mbq_results = run_mbq(
            vae_model=vae_model,
            model= model,
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
        
    # print(model)

    if pseudo_quant:
        mbq_results = torch.load(scale_path, map_location="cuda")
        apply_mbq(model, mbq_results)
        
        # original_state_dict = copy.deepcopy(model.state_dict())

        if not wa_quant:
            pseudo_quantize_model_weight(model, w_bit=w_bit, q_config=q_config)
        else:
            pseudo_quantize_model_weight_act(model, w_bit=w_bit, a_bit=a_bit)
    
#         # 获取量化后的模型权重
#         quantized_state_dict = model.state_dict()

#         # 计算每层的MSE
#         # 计算每层的误差指标并保存到文件
#         with open("/root/autodl-tmp/quantify/mse_results.txt", "a") as f:
#             print("\n=== Error Comparison of Parameters Before and After Quantization ===")
#             f.write("\n=== Error Comparison of Parameters Before and After Quantization ===\n")

#             total_mse = 0.0
#             total_mae = 0.0
#             total_max_error = 0.0
#             total_relative_error = 0.0
#             param_count = 0

#             for key in original_state_dict:
#                 if 'weight' in key:  # 只比较权重参数
#                     orig_weight = original_state_dict[key]
#                     quant_weight = quantized_state_dict[key]

#                     # 计算各指标
#                     mse = compute_mse(orig_weight, quant_weight)
#                     mae = compute_mae(orig_weight, quant_weight)
#                     max_error = compute_max_error(orig_weight, quant_weight)
#                     rel_error = compute_relative_error(orig_weight, quant_weight)

#                     # 累加总和
#                     total_mse += mse
#                     total_mae += mae
#                     total_max_error += max_error
#                     total_relative_error += rel_error
#                     param_count += 1

#                     # 打印和写入文件
#                     # print(f"Layer '{key}':")
#                     # print(f"  MSE: {mse:.6f}")
#                     # print(f"  MAE: {mae:.6f}")
#                     # print(f"  Max Absolute Error: {max_error:.6f}")
#                     # print(f"  Relative Error: {rel_error:.6f}")

#                     f.write(f"Layer '{key}': ")
#                     f.write(f"  MSE: {mse:.6f} ")
#                     f.write(f"  MAE: {mae:.6f} ")
#                     f.write(f"  Max Absolute Error: {max_error:.6f} ")
#                     f.write(f"  Relative Error: {rel_error:.6f}\n")

#             # 计算平均值
#             if param_count > 0:
#                 avg_mse = total_mse / param_count
#                 avg_mae = total_mae / param_count
#                 avg_max_error = total_max_error / param_count
#                 avg_relative_error = total_relative_error / param_count

#                 print(f"\nAverage across all weight layers:")
#                 print(f"  MSE: {avg_mse:.6f}")
#                 print(f"  MAE: {avg_mae:.6f}")
#                 print(f"  Max Absolute Error: {avg_max_error:.6f}")
#                 print(f"  Relative Error: {avg_relative_error:.6f}")

#                 f.write(f"\nAverage across all weight layers: ")
#                 f.write(f"  MSE: {avg_mse:.6f} ")
#                 f.write(f"  MAE: {avg_mae:.6f} ")
#                 f.write(f"  Max Absolute Error: {avg_max_error:.6f} ")
#                 f.write(f"  Relative Error: {avg_relative_error:.6f}\n")
#             else:
#                 print("No weight parameters found to compare.")
#                 f.write("No weight parameters found to compare.\n")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='/root/autodl-tmp/pretrained_models/var_d30.pth')
    parser.add_argument("--vae_ckpt", type=str, default='/root/autodl-tmp/pretrained_models/vae_ch160v4096z32.pth')
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--scale_path", type=str, default="/root/autodl-tmp/quantify/var_mbq_scales_16.pt")
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--a_bit", type=int, default=6)
    parser.add_argument("--important_scales", type=int, default=3)
    parser.add_argument('--output_dir', type=str, required=False, help="Output directory to save images",default="/root/autodl-tmp/outputs")
    parser.add_argument('--num_images', type=int, required=False, help="Number of images to generate",default=50000)
    parser.add_argument('--num_iter', type=int, required=False, help="Batch size per iteration",default=64)
    parser.add_argument('--wa_quant', type=bool, required=False, help="Batch size per iteration",default=True)
    args = parser.parse_args()
    

    print("Quantify-256-w_bit{}-a_bit{}-image{}-wa_quant {}".format(args.w_bit,args.a_bit,args.num_images,args.wa_quant))
    # 设置随机种子
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    output_dir = args.output_dir

    # 运行优化设置
    # tf32 = True
    # torch.backends.cudnn.allow_tf32 = bool(tf32)
    # torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    # torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # 初始化VAR模型
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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

    # 准备示例输入用于量化
    B = 250  # 小批量用于量化
    batch_size = 1000//B 
    
    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    
    val_aug = transforms.Compose([
        transforms.Resize(round(1.125 * 256), interpolation=InterpolationMode.LANCZOS),  # 调整图像大小
        transforms.CenterCrop((256, 256)),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        normalize_01_into_pm1,  # 归一化到 [-1, 1]
    ])
    
    
    parent_folder = "/root/autodl-tmp/val"  # 目标父文件夹路径，可以根据需要修改

    # 获取父文件夹下的所有子文件夹，并按名字升序排列
    subfolders = [f for f in os.listdir(parent_folder) 
                  if os.path.isdir(os.path.join(parent_folder, f))]
    subfolders.sort()  # 按名字升序排列
    
    # 检查子文件夹数量是否足够
    if len(subfolders) < 1000:
        raise ValueError(f"子文件夹数量 ({len(subfolders)}) 小于批量大小 B ({B})")
        
    subfolders = subfolders[::batch_size]

    # 初始化存储图片的列表
    inp_B3HW = []
    labels = list(range(B))  # 从0到999的标签列表

    # 从每个子文件夹中随机读取一张图片
    for i in range(B):
        folder = os.path.join(parent_folder, subfolders[i])
        # 获取文件夹中的所有图片文件
        image_files = [f for f in os.listdir(folder) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not image_files:
            print(f"警告: 文件夹 '{folder}' 中没有图片，跳过")
            continue

        # 随机挑选一张图片
        selected_file = random.choice(image_files)
        image_path = os.path.join(folder, selected_file)

        # 加载图片并应用预处理
        image = Image.open(image_path).convert('RGB')  # 加载图片并确保是 RGB 格式
        image = val_aug(image)  # 应用预处理
        inp_B3HW.append(image)
        
    # inp_B3HW = torch.stack(inp_B3HW).to(device)  # 将图片堆叠为张量并移动到设备
    # gt_idx_Bl = vae.img_to_idxBl(inp_B3HW) 
    # print("hellow",len(gt_idx_Bl))
    # gt_BL = torch.cat(gt_idx_Bl, dim=1)
    # x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # 生成模型输入

    # 分批处理VAE编码
    num_batches = (B + batch_size - 1) // batch_size  # 计算总批次数
    gt_idx_Bl_all = []
    x_BLCv_wo_first_l_all = []
    device = torch.device("cuda")

    print("Processing images through VAE...")
    for batch_idx in tqdm(range(num_batches), desc="VAE Encoding"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, B)

        # 提取当前批次的图片
        inp_batch = torch.stack(inp_B3HW[start_idx:end_idx]).to(device)

        # VAE编码
        gt_idx_Bl = vae.img_to_idxBl(inp_batch)  # 假设返回的是列表形式
        x_batch = vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        # 存储结果
        gt_idx_Bl_all.append(torch.cat(gt_idx_Bl, dim=1))  # 在批次内先按dim=0合并
        x_BLCv_wo_first_l_all.append(x_batch.cpu())  # 先移回CPU节省显存

        # 清理显存
        del inp_batch, gt_idx_Bl, x_batch
        torch.cuda.empty_cache()

    # 合并所有批次的结果
    gt_BL = torch.cat(gt_idx_Bl_all, dim=0).to(device)  # 按batch维度合并
    x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l_all, dim=0).to(device)
    
    # print(len(gt_BL),len(x_BLCv_wo_first_l))
    # print("hellow",gt_BL.size(), x_BLCv_wo_first_l.size())

    
    # print(x_BLCv_wo_first_l.shape)
    
    # labels = [278, 13, 16, 102, 264, 327, 540, 604]
    # expanded_labels = [label for label in labels for _ in range(1)]

    # 准备prompt_kwargs
    prompt_kwargs = {
        "label_B": torch.tensor(labels, device=device),  # 示例标签
        "x_BLCv_wo_first_l": x_BLCv_wo_first_l,  # 使用真实图片生成的数据
        "gt_BL": gt_BL,
    }
    
    # 获取mask
    important_mask, less_important_mask = get_scale_importance_masks(patch_nums, args.important_scales)
    
    # 生成 scale_masks 列表
    scale_masks = get_scale_masks(patch_nums)
    
    # 准备prompt_inputs
    prompt_inputs = {
        "vision_mask": important_mask,
        "caption_mask": less_important_mask,
        "scale_masks": scale_masks  # 传入所有mask的列表
    }

     # 计算 MBQ
    quantized_var = var_mbq_entry(
        vae_model=vae,
        model=var,
        inputs=prompt_kwargs,
        prompt_inputs=prompt_inputs,
        run_mbq_process=True,
        pseudo_quant=False,
        scale_path=args.scale_path,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        wa_quant=args.wa_quant,
        important_scales=args.important_scales,
        reweight=True,
        patch_nums=patch_nums,
        distort=False
    )

    # 第2步：删除模型并释放内存
    print("Step 2: Deleting models and clearing memory...")
    del vae, var, quantized_var
    torch.cuda.empty_cache()

    # 第3步：重新加载模型并应用伪量化
    print("Step 3: Reloading models and applying pseudo quantization...")
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
        run_mbq_process=False,  # 使用已保存的 MBQ 结果
        pseudo_quant=True,     # 应用伪量化
        scale_path=args.scale_path,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        wa_quant=args.wa_quant,
        important_scales=args.important_scales,
        reweight=True,
        patch_nums=patch_nums
    )
    
    quantized_var.to(device)
    quantized_var.eval()
    print(quantized_var)
    
    # set args
    num_images = args.num_images
    num_iter = args.num_iter
    batch_size = num_iter
    cfg = args.cfg
    output_dir = args.output_dir

    seed = 0  # Fixed seed for reproducibility
    torch.manual_seed(seed)
    num_sampling_steps = 250
    more_smooth = False
    
    save_folder = os.path.join(output_dir, "Quantify-256-w_bit{}-a_bit{}-image{}-wa_quant{}".format(args.w_bit,args.a_bit,num_images,args.wa_quant))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    num_steps = num_images // batch_size + 1

    class_name = 1000
    assert num_images % class_name == 0
    class_label_gen_world = np.arange(0, class_name).repeat(num_images // class_name)
    
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    used_time = 0.0
    gen_img_cnt = 0
    
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[batch_size * i:min(batch_size * (i + 1), num_images)]
        label_B: torch.LongTensor = torch.tensor(labels_gen, device=device)
        B = len(labels_gen)

        # Start timing
        torch.cuda.synchronize()
        event_start.record()

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = quantized_var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed, more_smooth=more_smooth)

        # End timing
        event_end.record()
        torch.cuda.synchronize()

        if i >= 1:
            elapsed_time = event_start.elapsed_time(event_end) / 1000.0  # Convert to seconds
            used_time += elapsed_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        # Normalize the tensor to [0, 1] range
        transformed_tensor = recon_B3HW.clamp(0, 1)  # Ensure values are in [0, 1]
        transformed_tensor = (transformed_tensor * 255).round().byte()  # Scale to [0, 255] and round

        # Pre-process all images into arrays
        image_arrays = [transformed_tensor[b_id].permute(1, 2, 0).cpu().numpy() for b_id in range(transformed_tensor.size(0))]

        # Convert arrays to images
        images = [Image.fromarray(img_array) for img_array in image_arrays]

        # Iterate over the images to save
        for b_id, img in enumerate(images):
            img_id = i * batch_size + b_id
            file_name = f"{str(img_id).zfill(5)}.png"
            img.save(os.path.join(save_folder, file_name))
        
        # break
            
    output_txt_path = os.path.join("/root/autodl-tmp/outputs", "results.txt")

    # 确保 output 文件夹存在
    os.makedirs("/root/autodl-tmp/outputs", exist_ok=True)
    
    torch.cuda.empty_cache()  # 清理缓存
    
    sec_per_image=used_time / gen_img_cnt
    write_save_folder_to_file(output_txt_path,save_folder,num_images,sec_per_image)
