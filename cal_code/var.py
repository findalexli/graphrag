import os
import json
import numpy as np

# 定义处理函数
def compute_recall_variances(input_dir, output_file):
    sample_variances = []

    # 遍历 0-999 的文件
    for index in range(1000):
        file_path = os.path.join(input_dir, f"best_graph_additional_{index}.json")
        if not os.path.exists(file_path):
            continue

        # 读取文件内容
        with open(file_path, 'r') as file:
            data = json.load(file)

        # 提取 recall 值
        recalls = [graph.get("recall", 0) for graph in data]
        # print(recalls)
        if len(recalls) > 0:
            # 计算每个样本的 recall 方差
            variance = np.var(recalls, ddof=1)  # 使用无偏方差
            sample_variances.append(variance)

    # 计算所有样本 recall 方差的方差
    overall_variance = np.var(sample_variances, ddof=1) if len(sample_variances) > 0 else 0

    # 写入结果到文件
    with open(output_file, 'w') as file:
        file.write("=== Recall Variances ===\n")
        for index, variance in enumerate(sample_variances):
            file.write(f"Sample {index}: Variance = {variance}\n")
        file.write("\n=== Overall Variance ===\n")
        file.write(f"Overall Variance of Recall Variances: {overall_variance}\n")

# 输入和输出路径
input_dir = './result/llm_3/'
output_file = './recall_variances.log'

# 调用处理函数
compute_recall_variances(input_dir, output_file)
