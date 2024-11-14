import os
import json
import numpy as np
from collections import defaultdict

def process_graph_files(input_dir, output_matrices_file, output_stats_file):
    matrix_size = 7
    all_matrices = {}  # 存储所有矩阵
    matrix_counts = defaultdict(int)  # 统计矩阵数量

    # 遍历所有文件
    for index in range(1000):
        file_path = os.path.join(input_dir, f"graph_attempts_all_{index}.json")
        if not os.path.exists(file_path):
            continue

        # 加载 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 遍历每个 graph
        for graph_index, graph_entry in enumerate(data):
            graph = graph_entry.get("graph", {}).get("graph", {})
            nodes = graph.get("nodes", [])

            # 跳过节点数超过 7 的图
            if len(nodes) > matrix_size:
                continue

            # 初始化 7x7 矩阵
            matrix = np.zeros((matrix_size, matrix_size), dtype=int)

            # 映射节点 ID 到矩阵索引
            node_map = {}
            for node in nodes:
                node_id = int(node["id"])
                node_map[node_id] = node_id - 1

                # 设置节点类型
                if node["node_type"] == "reasoning":
                    matrix[node_id - 1, node_id - 1] = 2
                elif node["node_type"] == "retrievalandreasoning":
                    matrix[node_id - 1, node_id - 1] = 1

            # 设置上游连接
            for node in nodes:
                node_id = int(node["id"])
                i = node_map[node_id]
                for upstream_id in node.get("upstream_node_ids", []):
                    j = node_map.get(int(upstream_id), None)
                    if j is not None:
                        matrix[i, j] = 1

            # 记录矩阵
            sample_id = f"{index}_{graph_index}"
            all_matrices[sample_id] = matrix
            matrix_counts[tuple(map(tuple, matrix))] += 1

    # 保存矩阵到文件
    with open(output_matrices_file, 'w') as file:
        file.write("=== All Matrices ===\n")
        for sample_id, matrix in all_matrices.items():
            file.write(f"Sample ID: {sample_id}\n")
            file.write(f"{np.array2string(matrix, separator=',')}\n\n")

    # 保存统计结果到文件
    with open(output_stats_file, 'w') as file:
        file.write("=== Matrix Statistics ===\n")
        for matrix, count in matrix_counts.items():
            matrix_array = np.array(matrix)
            file.write(f"Matrix:\n{np.array2string(matrix_array, separator=',')}\n")
            file.write(f"Count: {count}\n\n")

# 输入和输出文件路径
input_dir = './result/llm_3/'
output_matrices_file = './all_matrices.log'
output_stats_file = './matrix_statistics.log'

# 调用处理函数
process_graph_files(input_dir, output_matrices_file, output_stats_file)
