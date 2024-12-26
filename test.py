# from npzmaker import create_npz_from_sample_folder

# save_folder="/root/autodl-tmp/outputs/256-ariter128-diffsteps250-cfg1.5-image500001"
# num_images=50000

# # Create npz file and clean up
# create_npz_from_sample_folder(save_folder, num_images)
# os.system(f"rm -r {save_folder}")
# os.system(f"python evaluator.py /root/autodl-tmp/pretrained_models/VIRTUAL_imagenet256_labeled.npz f{save_folder}.npz")

from calflops import calculate_flops
import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import time
import argparse
from PIL import Image

def compute_model_flops(model, input_shape):
    """
    Compute the FLOPs, MACs, and Params of a given model.

    Args:
        model: PyTorch model to evaluate.
        input_shape: Tuple indicating the input shape, e.g., (batch_size, channels, height, width).

    Returns:
        A dictionary containing FLOPs, MACs, and Params.
    """
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4
    )
    print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")
    return {"FLOPs": flops, "MACs": macs, "Params": params}

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}

# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = '/root/autodl-tmp/pretrained_models/vae_ch160v4096z32.pth', f"/root/autodl-tmp/pretrained_models/var_d{MODEL_DEPTH}.pth"
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

# Compute and print model FLOPs
input_shape = (1, 3, 224, 224)  # Example input shape
print("Calculating FLOPs for VAE model...")
vae_flops = compute_model_flops(vae, input_shape=(1, 3, 224, 224))
print("VAE FLOPs:", vae_flops)

print("Calculating FLOPs for VAR model...")
var_flops = compute_model_flops(var, input_shape=(1, 32, 16, 16))  # Adjust input shape as per VAR input
print("VAR FLOPs:", var_flops)
