import json
import os

# 文件路径
input_path = "./result/llm_4"
output_file = "./analyse_result/new_system/merged_lowest_graphs.json"

# 初始化列表
combined_graphs = []

# 逐个读取文件
for i in range(1000):  # 假设index范围为0-999
    file_name = f"lowest_graph_additional_{i}.json"
    file_path = os.path.join(input_path, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # 读取并解析JSON文件
    with open(file_path, "r") as f:
        graphs = json.load(f)
        combined_graphs.extend(graphs)  # 添加到列表中

# 将合并后的内容写入新的JSON文件
with open(output_file, "w") as f:
    json.dump(combined_graphs, f, indent=4)

print(f"Combined JSON saved to: {output_file}")
