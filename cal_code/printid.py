# import os
# import json

# # 设定目标目录
# directory = './result/llm_3/'

# # 遍历 0 到 999 的文件
# for index in range(1000):
#     filename = f'graph_attempts_all_{index}.json'
#     filepath = os.path.join(directory, filename)

#     # 检查文件是否存在
#     if os.path.exists(filepath):
#         with open(filepath, 'r', encoding='utf-8') as file:
#             try:
#                 data = json.load(file)
#                 # 遍历每个图的数据
#                 for item in data:
#                     nodes = item.get('graph', {}).get('graph', {}).get('nodes', [])
#                     # 打印所有节点的 id
#                     for node in nodes:
#                         print(node.get('id'))
#             except json.JSONDecodeError as e:
#                 print(f"Error decoding JSON in file {filepath}: {e}")
#     else:
#         print(f"File not found: {filepath}")


# 读取并显示./output_combined中的all_graphs.npy文件

import numpy as np

# 读取文件
all_graphs = np.load('./output_combined/all_graphs.npy', allow_pickle=True)

# 打印第一个元素
print(all_graphs[0])

# 打印第一个元素的哈希值
print(all_graphs[0]['hash'])

# 打印第一个元素的矩阵
print(all_graphs[0]['matrix'])

# 打印第一个元素的矩阵的形状
print(all_graphs[0]['matrix'].shape)

# 打印第一个元素的矩阵的数据类型
print(all_graphs[0]['matrix'].dtype)

# 打印第一个元素的矩阵的数据
print(all_graphs[0]['matrix'].data)
# 请注意，这里的数据是一个内存地址，而不是实际的数据。
