import os
import subprocess
import time
import torch

# 定义参数组合
bit_combinations = [
    (6, 6),
    (4, 8),
    (8, 4),
    (4, 6),
    (6, 4),
]

wa_quant_values = [True]

# 日志文件路径
log_file = "../log.txt"

# 确保日志目录存在
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 遍历所有组合并运行
for w_bit, a_bit in bit_combinations:
    for wa_quant in wa_quant_values:
        
        torch.cuda.empty_cache()
        # 构造命令 - 修正了空格问题
        command = [
            "python",
            "script-quantify.py",
            "--w_bit", str(w_bit),
            "--a_bit", str(a_bit),
            "--wa_quant", str(wa_quant).lower(),
        ]

        # 写入日志（使用追加模式自动处理文件创建）
        with open(log_file, "a") as f:
            f.write(f"Running command: {' '.join(command)}\n")

        try:
            start_time = time.time()
            subprocess.run(command, check=True)
            end_time = time.time()
            
            # 写入成功信息
            with open(log_file, "a") as f:
                f.write(f"Completed: w_bit={w_bit}, a_bit={a_bit}, wa_quant={wa_quant}\n")
                f.write(f"Time taken: {end_time - start_time:.2f} seconds\n\n")
        except subprocess.CalledProcessError as e:
            # 写入错误信息
            with open(log_file, "a") as f:
                f.write(f"Error occurred for w_bit={w_bit}, a_bit={a_bit}, wa_quant={wa_quant}\n")
                f.write(f"Error: {str(e)}\n\n")
            continue

# 写入最终完成信息
with open(log_file, "a") as f:
    f.write("All experiments completed!\n")