import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import time
import argparse
from PIL import Image
from test import write_save_folder_to_file

import os
import torch

def load_var_model(vae_ckpt_path: str, var_ckpt_path: str, model_depth: int, device: str, patch_nums: tuple):
    """
    加载指定深度的VAR模型。
    
    :param vae_ckpt_path: VQVAE模型的路径
    :param var_ckpt_path: VAR模型的路径
    :param model_depth: VAR模型的深度
    :param device: 设备类型 (cuda or cpu)
    :return: 加载的VQVAE和VAR模型
    """
    from models import build_vae_var  # 需要导入VQVAE和VAR构建函数

    # VQVAE的硬编码超参数
    vae_args = {
        'V': 4096, 'Cvae': 32, 'ch': 160, 'share_quant_resi': 4,
        'device': device, 'patch_nums': patch_nums,
        'num_classes': 1000, 'depth': model_depth, 'shared_aln': False
    }

    # 加载VQVAE和VAR模型
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, device=device,
        patch_nums=patch_nums, num_classes=1000, depth=model_depth
    )

    # 加载模型权重
    vae.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt_path, map_location='cpu'), strict=True)

    vae.eval()
    var.eval()

    # 禁用梯度计算
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)

    print(f'Model loaded: VAE and VAR depth={model_depth}')
    
    return vae, var


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Sample images with VAE-VAR model")
    parser.add_argument('--num_images', type=int, required=False, help="Number of images to generate",default=5000)
    parser.add_argument('--num_iter', type=int, required=False, help="Batch size per iteration",default=64)
    parser.add_argument('--cfg', type=float, required=False, help="Classifier-free guidance scale",default=1.5)
    parser.add_argument('--output_dir', type=str, required=False, help="Output directory to save images",default="/root/autodl-tmp/outputs")
    parser.add_argument('--model_depth_1', type=int, required=False,default=30)
    parser.add_argument('--model_depth_2', type=int, required=False,default=30)
    args = parser.parse_args()

    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
    from models import VQVAE, build_vae_var

    MODEL_DEPTH1 = args.model_depth_1    
    MODEL_DEPTH2 = args.model_depth_2    
    # TODO: =====> please specify MODEL_DEPTH <=====
    # assert MODEL_DEPTH in {16, 20, 24, 30}

    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt = '/root/autodl-tmp/pretrained_models/vae_ch160v4096z32.pth'
    var_ckpt1 = f"/root/autodl-tmp/pretrained_models/var_d{MODEL_DEPTH1}.pth"
    var_ckpt2 = f"/root/autodl-tmp/pretrained_models/var_d{MODEL_DEPTH2}.pth"
    if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt1): os.system(f'wget {hf_home}/{var_ckpt1}')
    if not osp.exists(var_ckpt2): os.system(f'wget {hf_home}/{var_ckpt2}')

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vae, var1 = load_var_model(vae_ckpt, var_ckpt1, MODEL_DEPTH1, device, patch_nums)
    vae, var2 = load_var_model(vae_ckpt, var_ckpt2, MODEL_DEPTH2, device, patch_nums)

    # set args
    num_images = args.num_images
    num_iter = args.num_iter
    batch_size = num_iter
    cfg = args.cfg
    output_dir = args.output_dir

    seed = 0  # Fixed seed for reproducibility
    torch.manual_seed(seed)
    num_sampling_steps = 250
    class_labels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 980, 980, 437, 437, 22, 22, 562, 562)
    more_smooth = False

    save_folder = os.path.join(output_dir, "256-depth1_{}-depth2_{}-ariter{}-diffsteps{}-cfg{}-image{}".format(MODEL_DEPTH1, MODEL_DEPTH2, num_iter, num_sampling_steps, cfg, num_images))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    num_steps = num_images // batch_size + 1

    class_name = 1000
    assert num_images % class_name == 0
    class_label_gen_world = np.arange(0, class_name).repeat(num_images // class_name)

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    used_time = 0.0
    gen_img_cnt = 0

    # sample
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[batch_size * i:min(batch_size * (i + 1), num_images)]
        label_B: torch.LongTensor = torch.tensor(labels_gen, device=device)
        B = len(labels_gen)

        # Start timing
        torch.cuda.synchronize()
        event_start.record()
        
        f_hat, next_token_map, cur_L=None, None, None

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                f_hat, next_token_map, cur_L = var1.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed, more_smooth=more_smooth, is_return_raw=True)
                print("stop")
                recon_B3HW = var1.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed, more_smooth=more_smooth, is_return_raw=False, f_hat_past=f_hat, next_token_map_past=next_token_map,cur_L_past=cur_L)
                
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
            
    output_txt_path = os.path.join("/root/autodl-tmp/outputs", "results.txt")

    # 确保 output 文件夹存在
    os.makedirs("/root/autodl-tmp/outputs", exist_ok=True)

    sec_per_image=used_time / gen_img_cnt
    write_save_folder_to_file(output_txt_path,save_folder,num_images,sec_per_image)
    
    
if __name__ == "__main__":
    main()
