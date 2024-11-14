import json
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import hashlib
from typing import Tuple
import numba
import itertools

def real_id(s: str) -> int:
    s = s.replace('node', '')
    s = s.replace('n', '')
    s = s.replace('N', '')
    s = s.replace('resoig_', '')
    return int(s)
@numba.njit
def compute_permuted_matrix(adj_matrix, perm):
    n = len(adj_matrix)
    permuted_matrix = np.empty_like(adj_matrix)
    for i in range(n):
        for j in range(n):
            permuted_matrix[i, j] = adj_matrix[perm[i], perm[j]]
    return permuted_matrix

def adjacency_matrix_hash(adj_matrix):
    graph = nx.from_numpy_array(adj_matrix)
    hash = nx.weisfeiler_lehman_graph_hash(graph)
    return adj_matrix, hash
import tqdm
# def main():
#     datas = os.listdir('result/llm_3')
#     original = {}
#     count = {}
#     for data in tqdm.tqdm(datas):
#         if not 'graph_attempts' in data:
#             continue
#         graphs = json.load(open(f'result/llm_3/{data}', 'r'))
#         # print(data)
#         # if only_best:
#         #     graphs = list(filter(lambda graph: graph['graph']['graph']['is_best'], graphs))
#         invalid = False
#         for graph in graphs:
#             # print(graph)
#             nodes = graph['graph']['graph']['nodes']
#             adj = np.zeros((15, 15))
            
#             for node in nodes:
#                 if 'id' not in node:
#                     print(node)
#                     node['id'] = node['node_id']
                
#                 self_id = real_id(node['id'])
#                 if self_id >= 15:
#                     invalid = True
#                     break
#                 for source in node['upstream_node_ids']:
#                     source = int(source)
#                     if source >= 8:
#                         invalid = True
#                         break
#                     adj[source, self_id] = 1
#                 if node['node_type'] == 'retrievalandreasoning':
#                     adj[self_id, self_id] = 1
#                 elif node['node_type'] == 'reasoning':
#                     adj[self_id, self_id] = 2
#                 if invalid:
#                     break
#             if not invalid:
#                 sadj, hash = adjacency_matrix_hash(adj)
#                 count[hash] = count.get(hash, 0) + 1
#                 original[hash] = sadj
    

# if __name__ == "__main__":
#     main()


import json
import os
import numpy as np
import tqdm

# def main():
#     datas = os.listdir('result/llm_3')
#     original = {}
#     count = {}
    
#     # 创建输出目录
#     output_dir = "output_json"
#     os.makedirs(output_dir, exist_ok=True)
    
#     for data in tqdm.tqdm(datas):
#         if not 'graph_attempts' in data:
#             continue
#         graphs = json.load(open(f'result/llm_3/{data}', 'r'))
#         invalid = False
#         for graph in graphs:
#             nodes = graph['graph']['graph']['nodes']
#             adj = np.zeros((15, 15))
            
#             for node in nodes:
#                 if 'id' not in node:
#                     node['id'] = node['node_id']
                
#                 self_id = real_id(node['id'])
#                 if self_id >= 15:
#                     invalid = True
#                     break
#                 for source in node['upstream_node_ids']:
#                     source = int(source)
#                     if source >= 8:
#                         invalid = True
#                         break
#                     adj[source, self_id] = 1
#                 if node['node_type'] == 'retrievalandreasoning':
#                     adj[self_id, self_id] = 1
#                 elif node['node_type'] == 'reasoning':
#                     adj[self_id, self_id] = 2
#                 if invalid:
#                     break
#             if not invalid:
#                 sadj, hash = adjacency_matrix_hash(adj)
#                 count[hash] = count.get(hash, 0) + 1
#                 original[hash] = sadj.tolist()  # 转为列表以便序列化为 JSON
                
#                 # 保存到单独的 JSON 文件
#                 output_path = os.path.join(output_dir, f"{hash}.json")
#                 with open(output_path, 'w') as f:
#                     json.dump({
#                         "hash": hash,
#                         "matrix": sadj.tolist()  # 将矩阵保存为列表
#                     }, f, indent=4)
    
#     # 保存总体统计结果
#     with open(os.path.join(output_dir, "summary.json"), 'w') as f:
#         json.dump({
#             "count": count,  # 记录每种图的出现次数
#             "original_hashes": list(original.keys())  # 所有哈希值
#         }, f, indent=4)

# if __name__ == "__main__":
#     main()
import json
import os
import numpy as np
import tqdm

# def main():
#     datas = os.listdir('result/llm_3')
#     count = {}
    
#     # 创建输出目录
#     output_dir = "output_json"
#     os.makedirs(output_dir, exist_ok=True)
    
#     for data in tqdm.tqdm(datas):
#         if not 'graph_attempts' in data:
#             continue
        
#         # 读取当前文件
#         graphs = json.load(open(f'result/llm_3/{data}', 'r'))
#         invalid = False
#         result = []  # 存储当前文件所有有效矩阵的结果
        
#         for graph in graphs:
#             nodes = graph['graph']['graph']['nodes']
#             adj = np.zeros((15, 15))
            
#             for node in nodes:
#                 if 'id' not in node:
#                     node['id'] = node['node_id']
                
#                 self_id = real_id(node['id'])
#                 if self_id >= 15:
#                     invalid = True
#                     break
#                 for source in node['upstream_node_ids']:
#                     source = int(source)
#                     if source >= 8:
#                         invalid = True
#                         break
#                     adj[source, self_id] = 1
#                 if node['node_type'] == 'retrievalandreasoning':
#                     adj[self_id, self_id] = 1
#                 elif node['node_type'] == 'reasoning':
#                     adj[self_id, self_id] = 2
#                 if invalid:
#                     break
            
#             # 如果图有效，保存邻接矩阵和哈希值
#             if not invalid:
#                 sadj, hash = adjacency_matrix_hash(adj)
#                 count[hash] = count.get(hash, 0) + 1
#                 result.append({
#                     "hash": hash,
#                     "matrix": sadj.tolist()
#                 })
        
#         # 将当前文件所有图的信息保存为一个 JSON 文件
#         output_path = os.path.join(output_dir, data)  # 文件名与原始文件相同
#         with open(output_path, 'w') as f:
#             json.dump(result, f, indent=4)
    
#     # 保存总体统计结果
#     with open(os.path.join(output_dir, "summary.json"), 'w') as f:
#         json.dump({
#             "count": count,  # 记录每种图的出现次数
#         }, f, indent=4)

# if __name__ == "__main__":
#     main()

import json
import os
import numpy as np
import tqdm

def main():
    datas = os.listdir('result/llm_3')
    count = {}
    all_results = []  # 用于存储所有矩阵和哈希值
    summary = {}      # 用于保存统计信息
    
    # 创建输出目录
    output_dir = "output_combined"
    os.makedirs(output_dir, exist_ok=True)
    
    for data in tqdm.tqdm(datas):
        if not 'graph_attempts' in data:
            continue
        
        # 读取当前文件
        graphs = json.load(open(f'result/llm_3/{data}', 'r'))
        invalid = False
        
        for graph in graphs:
            nodes = graph['graph']['graph']['nodes']
            adj = np.zeros((15, 15))
            
            for node in nodes:
                if 'id' not in node:
                    node['id'] = node['node_id']
                
                self_id = real_id(node['id'])
                if self_id >= 15:
                    invalid = True
                    break
                for source in node['upstream_node_ids']:
                    source = int(source)
                    if source >= 8:
                        invalid = True
                        break
                    adj[source, self_id] = 1
                if node['node_type'] == 'retrievalandreasoning':
                    adj[self_id, self_id] = 1
                elif node['node_type'] == 'reasoning':
                    adj[self_id, self_id] = 2
                if invalid:
                    break
            
            # 如果图有效，保存邻接矩阵和哈希值
            if not invalid:
                sadj, hash = adjacency_matrix_hash(adj)
                count[hash] = count.get(hash, 0) + 1
                all_results.append({
                    "hash": hash,
                    "matrix": sadj
                })
    
    # 保存所有矩阵和哈希值到一个 .npy 文件
    combined_path = os.path.join(output_dir, "all_graphs.npy")
    np.save(combined_path, all_results)  # 保存整个列表到 .npy 文件
    
    # 保存 summary 信息到 JSON 文件
    summary["count"] = count
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()

