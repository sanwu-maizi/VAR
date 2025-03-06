import os

# 定义参数
num_iter = 50000
model_depth_list = [24]
tome_scale_list = [0.1,0.2]

# 循环调用脚本
for tome_scale in tome_scale_list:
    command = f"python script-256.py --num_images {num_iter} --tome_scale {tome_scale}"
    print(f"Executing: {command}")
    os.system(command)  # 调用命令
