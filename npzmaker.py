from tqdm import tqdm
import numpy as np
from PIL import Image
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:05d}.png") # 修改名称
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3) 
    npz_path = f"{sample_dir}.npz"
    np.savez(sample_dir, arr_0=samples)
    print(f"Saved .npz file to {sample_dir} [shape={samples.shape}].")
    return npz_path

sample_folder_dir = "/root/autodl-tmp/outputs/256-ariter128-diffsteps250-cfg1.5-image500001"
img_num = 50000 # 修改总数目
create_npz_from_sample_folder(sample_folder_dir, img_num)
