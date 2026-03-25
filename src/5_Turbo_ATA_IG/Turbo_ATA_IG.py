import numpy as np
import pandas as pd
import time
import random
import math
from numba import njit


# ==========================================
# 1. 极限加速模块 (直接翻译为底层 C 机器码)
# ==========================================
@njit
def calc_makespan_numba(seq, p):
    n = len(seq)
    m = p.shape[1]
    C = np.zeros(m)
    for i in range(n):
        job_idx = seq[i]
        C[0] += p[job_idx, 0]
        for j in range(1, m):
            if C[j] > C[j - 1]:
                C[j] = C[j] + p[job_idx, j]
            else:
                C[j] = C[j - 1] + p[job_idx, j]
    return C[-1]


@njit
def taillard_insert_eval_numba(seq, job, p):
    n = len(seq)
    m = p.shape[1]

    # 前向矩阵
    e = np.zeros((n + 2, m + 1))
    for i in range(1, n + 1):
        job_idx = seq[i - 1]
        for j in range(1, m + 1):
            if e[i, j - 1] > e[i - 1, j]:
                e[i, j] = e[i, j - 1] + p[job_idx, j - 1]
            else:
                e[i, j] = e[i - 1, j] + p[job_idx, j - 1]

    # 后向矩阵
    q = np.zeros((n + 2, m + 2))
    for i in range(n, 0, -1):
        job_idx = seq[i - 1]
        for j in range(m, 0, -1):
            if q[i, j + 1] > q[i + 1, j]:
                q[i, j] = q[i, j + 1] + p[job_idx, j - 1]
            else:
                q[i, j] = q[i + 1, j] + p[job_idx, j - 1]

    best_ms = 1e15
    best_pos = -1
    C = np.zeros(m + 1)

    # 极速评估所有插入点
    for k in range(n + 1):
        ms = 0.0
        for j in range(1, m + 1):
            if C[j - 1] > e[k, j]:
                C[j] = C[j - 1] + p[job, j - 1]
            else:
                C[j] = e[k, j] + p[job, j - 1]
            total_time = C[j] + q[k + 1, j]
            if total_time > ms:
                ms = total_time
        if ms < best_ms:
            best_ms = ms
            best_pos = k

    return best_ms, best_pos


# ==========================================
# 2. 涡轮版 ATA-IG 主程序 (含自适应破局机制)
# ==========================================
if __name__ == "__main__":
    matrix_file = '../../data/extracted_JSSP_data.csv'
    sequence_file = '../../result/truncated_neh_taillard_.csv'

    # --- SOTA 黄金参数 ---
    BASE_D_SIZE = 1  # 常规破坏尺度 d=1 (高频微调)
    KICK_D_SIZE = 4  # 破局破坏尺度 d=4 (强力电击)
    KICK_THRESHOLD = 2000  # 连续多少次不破纪录就触发电击
    TEMP_FACTOR = 2.0  # 温和退火
    TIME_LIMIT = 1800  # 运行限时 (秒)

    random.seed(42)
    np.random.seed(42)

    print("正在加载数据与 Numba JIT 预热...")
    times_data = pd.read_csv(matrix_file, header=None).values.astype(np.float64)
    df_seq = pd.read_csv(sequence_file)
    init_sequence = df_seq['Job_Order'].values.astype(np.int64)
    n, m = times_data.shape

    # JIT 预热 (第一次调用 Numba 函数会花零点几秒编译)
    _ = calc_makespan_numba(init_sequence[:2], times_data)
    _ = taillard_insert_eval_numba(init_sequence[:2], np.int64(init_sequence[2]), times_data)

    current_seq = init_sequence.copy()
    current_ms = calc_makespan_numba(current_seq, times_data)
    best_seq = current_seq.copy()
    best_ms = current_ms

    print("\n" + "🚀" * 20)
    print(" 启动 Turbo-ATA-IG (终极自适应破局版)")
    print(f" 初始基座 Makespan: {best_ms:.2f}")
    print(f" 比赛限时: {TIME_LIMIT} 秒！")
    print("🚀" * 20 + "\n")

    total_p_time = np.sum(times_data)
    temperature = TEMP_FACTOR * (total_p_time / (n * m * 10))

    start_time = time.time()
    iteration = 0
    sa_accepts = 0
    stagnation_counter = 0  # 🌟 新增：停滞计数器

    while time.time() - start_time < TIME_LIMIT:
        iteration += 1
        temp_seq = current_seq.copy()

        # ==========================================
        # 🌟 自适应破坏尺度监控
        # ==========================================
        current_d = BASE_D_SIZE
        if stagnation_counter > KICK_THRESHOLD:
            current_d = KICK_D_SIZE
            stagnation_counter = 0  # 触发后重置计数器
            # 可以在这里取消注释以观察破局触发频率
            # print(f"  [!] 停滞过久，触发强力扰动 (d={KICK_D_SIZE})...")

        # Phase 1: 破坏阶段 (拔出 current_d 个工件)
        removed_jobs = []
        for _ in range(current_d):
            idx_to_remove = random.randrange(len(temp_seq))
            removed_jobs.append(temp_seq[idx_to_remove])
            temp_seq = np.delete(temp_seq, idx_to_remove)

        # Phase 2: 重建阶段 (利用 Numba-Taillard 极速插回)
        new_ms = 0
        for job in removed_jobs:
            best_ins_ms, best_pos = taillard_insert_eval_numba(temp_seq, np.int64(job), times_data)
            temp_seq = np.insert(temp_seq, best_pos, job)
            new_ms = best_ins_ms

        # Phase 3: Metropolis 退火接受准则
        if new_ms < current_ms:
            current_seq = temp_seq.copy()
            current_ms = new_ms

            # 检查是否打破了【全局最优】纪录
            if new_ms < best_ms:
                best_ms = new_ms
                best_seq = temp_seq.copy()
                stagnation_counter = 0  # 🌟 破纪录了！停滞清零
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.2f}s | 迭代:{iteration}] ⚡ 破纪录! 新 Makespan: {best_ms:.2f}")
            else:
                stagnation_counter += 1  # 解变好了，但没破全局纪录，继续累加停滞
        else:
            stagnation_counter += 1  # 解变差了，累加停滞
            delta = new_ms - current_ms
            prob = math.exp(-delta / temperature)
            if random.random() <= prob:
                current_seq = temp_seq.copy()
                current_ms = new_ms
                sa_accepts += 1

    print("\n" + "👑" * 15)
    print(f" 时间到！总计执行了 {iteration} 次全量迭代！")
    print(f" 最终极光 Makespan: {best_ms:.2f}")
    print("👑" * 15 + "\n")

    pd.DataFrame({'Job_Order': best_seq}).to_csv(f'Turbo_ATA_IG_{int(best_ms)}.csv', index=False)


#  [1673.07s | 迭代:94360] ⚡ 破纪录! 新 Makespan: 138571.58
# [1673.18s | 迭代:94366] ⚡ 破纪录! 新 Makespan: 138571.58
# [1673.28s | 迭代:94372] ⚡ 破纪录! 新 Makespan: 138571.58

# 时间到！总计执行了 101474 次全量迭代！
 #最终极光 Makespan: 138571.58