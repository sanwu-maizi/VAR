################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import time
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
from PIL import Image
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = '/root/autodl-tmp/vae_ch160v4096z32.pth', f'/root/autodl-tmp/var_d{MODEL_DEPTH}.pth'
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


############################# 2. Sample with classifier-free guidance

# set args
num_images=50000
num_iter=16
batch_size=num_iter
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 1.5 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (0,1,2,3,4,5,6,7,8,9,10,980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
more_smooth = False # True for more smooth output

output_dir="/root/autodl-tmp/VAR-main/outputs"
save_folder = os.path.join(output_dir, "256-ariter{}-diffsteps{}-cfg{}-image{}".format(num_iter,num_sampling_steps,cfg,num_images))
if not os.path.exists(save_folder):
            os.makedirs(save_folder)
num_steps=num_images//batch_size+1

class_name=1000
assert num_images%class_name==0
class_label_gen_world=np.arange(0,class_name).repeat(num_images//class_name)


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
used_time=0.0
gen_img_cnt=0

# sample
for i in range(num_steps):
    print("Generation step {}/{}".format(i, num_steps))

    labels_gen=class_label_gen_world[batch_size * i:min(batch_size * (i+1),num_images)]
    label_B: torch.LongTensor = torch.tensor(labels_gen, device=device)
    B = len(labels_gen)
    
    
    # Start timing
    torch.cuda.synchronize()
    event_start.record()
    
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed, more_smooth=more_smooth)
            
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

