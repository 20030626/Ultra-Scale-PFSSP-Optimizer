import pandas as pd
import numpy as np
import time
import random
import math
import datetime


# ==========================================
# 1. 数据加载
# ==========================================
def load_data(file_path):
    return pd.read_csv(file_path, header=None).values


def load_initial_sequence(file_path):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 加载初始 SOTA 序列: {file_path}...")
    df = pd.read_csv(file_path)
    return df['Job_Order'].tolist()


def calculate_makespan(permutation, times_matrix):
    m = times_matrix.shape[1]
    if len(permutation) == 0: return 0
    completion_times = np.zeros(m)
    for job_idx in permutation:
        completion_times[0] += times_matrix[job_idx, 0]
        for mach_idx in range(1, m):
            completion_times[mach_idx] = max(completion_times[mach_idx - 1], completion_times[mach_idx]) + times_matrix[
                job_idx, mach_idx]
    return completion_times[-1]


# ==========================================
# 2. Taillard 加速引擎
# ==========================================
def find_best_insertion_taillard(current_seq, new_job, times_matrix):
    k = len(current_seq)
    m = times_matrix.shape[1]

    e = np.zeros((k + 1, m))
    for i in range(1, k + 1):
        job_idx = current_seq[i - 1]
        e[i, 0] = e[i - 1, 0] + times_matrix[job_idx, 0]
        for j in range(1, m):
            e[i, j] = max(e[i, j - 1], e[i - 1, j]) + times_matrix[job_idx, j]

    q = np.zeros((k + 2, m))
    for i in range(k, 0, -1):
        job_idx = current_seq[i - 1]
        q[i, m - 1] = q[i + 1, m - 1] + times_matrix[job_idx, m - 1]
        for j in range(m - 2, -1, -1):
            q[i, j] = max(q[i, j + 1], q[i + 1, j]) + times_matrix[job_idx, j]

    best_ms = float('inf')
    best_pos = 0

    for pos in range(k + 1):
        f = np.zeros(m)
        f[0] = e[pos, 0] + times_matrix[new_job, 0]
        for j in range(1, m):
            f[j] = max(f[j - 1], e[pos, j]) + times_matrix[new_job, j]

        current_max = 0
        for j in range(m):
            if f[j] + q[pos + 1, j] > current_max:
                current_max = f[j] + q[pos + 1, j]

        if current_max < best_ms:
            best_ms = current_max
            best_pos = pos

    return best_pos, best_ms


# ==========================================
# 3. Hybrid 3.0: 纯正 Taillard-SA-IG 终极微调
# ==========================================
def run_taillard_sa_ig(times_matrix, initial_seq, time_limit=300, d=4, temp_factor=0.4):
    n, m = times_matrix.shape
    print(f"\n--- 启动 Hybrid 3.0: Taillard 极速 SA-IG (限时 {time_limit} 秒, d={d}) ---")

    current_seq = initial_seq.copy()
    current_ms = calculate_makespan(current_seq, times_matrix)
    print(f"初始输入 Makespan: {current_ms:.2f} (起点)")

    best_seq = current_seq.copy()
    best_ms = current_ms

    # 动态计算自适应温度 (Ruiz & Stutzle 2007 官方设定)
    total_p_time = np.sum(times_matrix)
    temperature = temp_factor * (total_p_time / (n * m * 10))
    print(f"系统自适应退火温度: {temperature:.2f}")

    start_time = time.time()
    iteration = 0
    improvements = 0
    sa_accepts = 0  # 记录接受劣解的次数

    while time.time() - start_time < time_limit:
        iteration += 1
        temp_seq = current_seq.copy()

        # ---------------------------------------------
        # Phase 1: 破坏阶段 (Destruction - 随机拔出 d 个工件)
        # ---------------------------------------------
        removed_jobs = []
        for _ in range(d):
            idx_to_remove = random.randrange(len(temp_seq))
            removed_jobs.append(temp_seq.pop(idx_to_remove))

        # ---------------------------------------------
        # Phase 2: 重建阶段 (Construction - 用 Taillard 极速插回)
        # ---------------------------------------------
        new_ms = 0
        for job in removed_jobs:
            best_pos, new_ms = find_best_insertion_taillard(temp_seq, job, times_matrix)
            temp_seq.insert(best_pos, job)
            # 注意：最后一次插入返回的 new_ms 就是整个新序列的真实 Makespan！
            # 这一步彻底省去了额外重新计算全量 Makespan 的灾难开销。

        # ---------------------------------------------
        # Phase 3: 接受准则 (SA Acceptance Criterion)
        # ---------------------------------------------
        if new_ms < current_ms:
            # 严格变好了：接受，并检查是否破纪录
            current_seq = temp_seq.copy()
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_seq = temp_seq.copy()
                improvements += 1
                print(f"[{time.time() - start_time:.1f}s] 迭代 {iteration} | 破纪录！新全局最优 Makespan: {best_ms:.2f}")
        else:
            # 变差了：启动模拟退火机制，给它一次“退一步海阔天空”的机会
            delta = new_ms - current_ms
            prob = math.exp(-delta / temperature)
            if random.random() <= prob:
                current_seq = temp_seq.copy()
                current_ms = new_ms
                sa_accepts += 1

    print(f"\n局部搜索结束！总探测次数: {iteration} | 全局破纪录: {improvements} 次 | 模拟退火接受劣解: {sa_accepts} 次")
    return best_seq, best_ms


# ==========================================
# 4. 执行入口
# ==========================================
if __name__ == "__main__":
    matrix_file = '../../data/extracted_JSSP_data.csv'
    # 请根据你实际的文件名修改
    sequence_file = '../../result/truncated_neh_taillard_.csv'

    try:
        times_data = load_data(matrix_file)
        init_sequence = load_initial_sequence(sequence_file)

        # 极限压榨 5 分钟 (300秒)
        final_seq, final_ms = run_taillard_sa_ig(times_data, init_sequence, time_limit=300)

        print("\n" + "★" * 50)
        print(f" 终局之战：Hybrid 3.0 (Taillard-SA-IG) 运行完毕！")
        print(f" 最终极光 Makespan: {final_ms:.2f}")
        print("★" * 50 + "\n")

        #output_filename = 'Final_Ultimate_Hybrid_3_Sequence.csv'
        #pd.DataFrame({'Job_Order': final_seq}).to_csv(output_filename, index=False)
        #print(f"✅ 终极 SOTA 序列已保存至: {output_filename}")

    except Exception as e:
        print(f"❌ 运行报错: {e}")
#Taillard 极速 SA-IG (限时 300 秒, d=4) ---

#初始输入 Makespan: 139095.58 (起点)

#系统自适应退火温度: 1.98

#[244.5s] 迭代 26 | 破纪录！新全局最优 Makespan: 139095.58

#局部搜索结束！总探测次数: 32 | 全局破纪录: 1 次 | 模拟退火接受劣解: 0 次
