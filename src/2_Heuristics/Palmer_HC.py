import pandas as pd
import numpy as np
import time
import random

random.seed(42)
# 1. 加载数据
df = pd.read_csv('../../data/extracted_JSSP_data.csv', header=None)
times = df.values


def calculate_makespan(permutation, times_matrix):
    """计算流水车间调度下的总完工时间(Makespan)"""
    n = len(permutation)
    m = times_matrix.shape[1]
    # completion_times[j] 记录当前工件在机器 j 上的完工时间
    completion_times = np.zeros(m)

    for job_idx in permutation:
        # 第一台机器的完工时间 = 上个工件完工时间 + 当前工件时间
        completion_times[0] += times_matrix[job_idx, 0]
        # 后面机器的完工时间取决于：1.本工件在前一台机完工 2.上个工件在本机完工
        for mach_idx in range(1, m):
            start_time = max(completion_times[mach_idx - 1], completion_times[mach_idx])
            completion_times[mach_idx] = start_time + times_matrix[job_idx, mach_idx]
    return completion_times[-1]


# 2. 启发式初始化: Palmer's Slope Index
# 逻辑：优先处理“前期快、后期慢”的工件，以减少后端机器空转
m = times.shape[1]
weights = np.array([-(m - (2 * j + 1)) for j in range(m)])
slope_indices = np.dot(times, weights)
best_perm = list(np.argsort(-slope_indices))  # 按指数降序排列
best_ms = calculate_makespan(best_perm, times)

# 3. 局部搜索优化 (Hill Climbing)
print(f"开始优化，初始 Makespan: {best_ms:.2f}")
start_time = time.time()
while time.time() - start_time < 1157.91 :  # 运行60秒
    # 随机尝试交换两个工件的入场顺序
    idx1, idx2 = random.sample(range(1000), 2)
    new_perm = best_perm.copy()
    new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]

    new_ms = calculate_makespan(new_perm, times)
    if new_ms < best_ms:  # 如果效果更好，则保留
        best_ms = new_ms
        best_perm = new_perm

print(f"优化后的最终 Makespan: {best_ms:.2f}")

# 4. 保存最优顺序
pd.DataFrame({'JobIndex': best_perm}).to_csv('../../result/Palmer_HC_best_sequence.csv', index=False)

#开始优化，初始 Makespan: 146367.86（原始的Palmer）
#（Palmer得到的结果后加上Hill Climbing）

#1157.91优化后的最终 Makespan: 143686.07

