import pandas as pd
import re
from pathlib import Path

# 输入文本文件路径和输出 Excel 文件路径
input_file = "./results-tome.txt"  # 替换为你的文本文件路径
output_file = "./output2.xlsx"  # 输出 Excel 文件路径

# 定义正则表达式，用于匹配每行数据
folder_pattern = r"save_folder: (.+?), (\d+\.\d+) sec per image"
metrics_pattern = r"Inception Score: (\d+\.\d+), FID: (\d+\.\d+), sFID: (\d+\.\d+), Precision: (\d+\.\d+), Recall: (\d+\.\d+)"

# 初始化数据存储
data = []

# 读取并解析文本文件
with open(input_file, "r") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        # 匹配 save_folder 行
        folder_match = re.match(folder_pattern, lines[i].strip())
        if folder_match:
            folder_name = folder_match.group(1)
            sec_per_image = float(folder_match.group(2))
            i += 1
            
            # 匹配下一行的指标
            if i < len(lines):
                metrics_match = re.match(metrics_pattern, lines[i].strip())
                if metrics_match:
                    inception_score = float(metrics_match.group(1))
                    fid = float(metrics_match.group(2))
                    sfid = float(metrics_match.group(3))
                    precision = float(metrics_match.group(4))
                    recall = float(metrics_match.group(5))
                    
                    # 将数据添加到列表
                    data.append({
                        "Save Folder": folder_name,
                        "Sec per Image": sec_per_image,
                        "Inception Score": inception_score,
                        "FID": fid,
                        "sFID": sfid,
                        "Precision": precision,
                        "Recall": recall
                    })
        i += 1

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存到 Excel 文件
df.to_excel(output_file, index=False)
print(f"Excel 文件已保存到: {output_file}")