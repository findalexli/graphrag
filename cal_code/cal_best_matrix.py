import os
import json
import numpy as np
from collections import defaultdict

# 定义处理函数
def process_graph_files(input_dir, output_log_path):
    matrix_size = 7
    matrices = {}
    matrix_counts = defaultdict(int)

    # 遍历 0-999 的文件
    for index in range(1000):
        file_path = os.path.join(input_dir, f"graph_attempts_all_{index}.json")
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r') as file:
            data = json.load(file)

        # 遍历 JSON 文件中的图
        for sample_id, entry in enumerate(data):
            if entry.get("is_best"):
                graph = entry.get("graph", {}).get("graph", {})
                nodes = graph.get("nodes", [])
                
                # 跳过节点数量超过 7 的图
                if len(nodes) > matrix_size:
                    continue
                
                # 初始化矩阵
                matrix = np.zeros((matrix_size, matrix_size), dtype=int)
                node_map = {}  # 用于将节点 ID 映射到矩阵索引
                
                # 设置节点属性
                for node in nodes:
                    node_id = int(node["id"])
                    node_map[node_id] = node_id - 1
                    node_type = node["node_type"]
                    
                    if node_type == "reasoning":
                        matrix[node_id - 1, node_id - 1] = 2
                    elif node_type == "retrievalandreasoning":
                        matrix[node_id - 1, node_id - 1] = 1
                
                # 设置上游连接
                for node in nodes:
                    node_id = int(node["id"])
                    i = node_map[node_id]
                    for upstream_id in node.get("upstream_node_ids", []):
                        j = node_map[int(upstream_id)]
                        matrix[i, j] = 1

                # 保存结果
                matrices[f"{index}_{sample_id}"] = matrix
                matrix_counts[tuple(map(tuple, matrix))] += 1

    # 写入日志文件
    with open(output_log_path, 'w') as log_file:
        # 输出每个矩阵
        log_file.write("=== Matrices ===\n")
        for sample_id, matrix in matrices.items():
            log_file.write(f"Sample ID: {sample_id}\n")
            log_file.write(f"{np.array2string(matrix, separator=',')}\n\n")

        # 输出统计结果
        log_file.write("\n=== Matrix Counts ===\n")
        for matrix, count in matrix_counts.items():
            matrix_array = np.array(matrix)
            log_file.write(f"Matrix:\n{np.array2string(matrix_array, separator=',')}\n")
            log_file.write(f"Count: {count}\n\n")

# 输入和输出路径
input_dir = './result/llm_3/'
output_log_path = './results.log'

# 调用处理函数
process_graph_files(input_dir, output_log_path)
