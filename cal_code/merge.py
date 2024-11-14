import os
import json

# 输入文件目录
input_dir = './result/llm_3/'
# 输出合并后的文件路径
output_file = './merged_beste_graphs.json'

# 初始化一个空列表用于存储所有数据
merged_data = []

# 遍历文件名为 graph_attempts_all_{index}.json 的文件，index 从 0 到 999
for index in range(1000):
    file_path = os.path.join(input_dir, f"best_graph_additional_{index}.json")
    if not os.path.exists(file_path):
        continue  # 如果文件不存在，跳过

    # 打开文件并加载数据
    with open(file_path, 'r') as file:
        data = json.load(file)
        merged_data.extend(data)  # 将数据添加到合并列表中

# 将合并后的数据保存为一个新的 JSON 文件
with open(output_file, 'w') as file:
    json.dump(merged_data, file, indent=2)  # 使用缩进格式化输出

print(f"合并完成，合并后的文件保存为: {output_file}")
