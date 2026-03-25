import numpy as np
import pandas as pd
import time
import random
from numba import njit


# ---------------------------------------------------------
# 1. 加速计算模块 (Numba JIT)
# ---------------------------------------------------------
@njit
def calc_makespan(seq, p):
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
def taillard_insert_eval(seq, job, p):
    n = len(seq)
    m = p.shape[1]

    e = np.zeros((n + 2, m + 1))
    for i in range(1, n + 1):
        job_idx = seq[i - 1]
        for j in range(1, m + 1):
            if e[i, j - 1] > e[i - 1, j]:
                e[i, j] = e[i, j - 1] + p[job_idx, j - 1]
            else:
                e[i, j] = e[i - 1, j] + p[job_idx, j - 1]

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


# ---------------------------------------------------------
# 2. 算法主类 (NEH + SA-Tabu Memetic)
# ---------------------------------------------------------
class TabuSearchPFSSP:
    def __init__(self, data_path):
        print("正在加载流水车间数据...")
        self.p = pd.read_csv(data_path, header=None).values.astype(np.float64)
        self.n, self.m = self.p.shape
        print(f"数据加载完成: {self.n}工件 x {self.m}机器")

    def load_neh_sequence(self, neh_path):
        """🌟 新增：读取外部 NEH 算好的高质量初始解"""
        print(f"正在加载 NEH 初始序列文件: {neh_path}...")
        df_seq = pd.read_csv(neh_path)
        # 确保是 int64，并且是 0-based 索引
        return df_seq['Job_Order'].values.astype(np.int64)

    def optimize(self, initial_seq_path, max_iterations=500000, max_time_seconds=1800):
        start_time = time.time()

        # 1. 直接加载 NEH 序列作为初始解
        current_seq = self.load_neh_sequence(initial_seq_path)
        global_best_seq = current_seq.copy()

        current_ms = calc_makespan(current_seq, self.p)
        global_best_ms = current_ms

        print(f"🔥 初始解 (NEH 导入): {global_best_ms:.2f} (耗时: {time.time() - start_time:.2f}s)")

        dummy_seq = np.array([0], dtype=np.int64)
        _ = taillard_insert_eval(dummy_seq, np.int64(1), self.p[:2, :2])

        tabu_list = np.zeros(self.n, dtype=np.int64)
        candidate_size = 8
        temperature = 2.5

        iteration = 0
        search_start = time.time()

        stagnation_counter = 0
        stagnation_limit = 500

        print(f"🚀 开始 SA-Tabu 极限寻优 (从 {global_best_ms:.2f} 往下砸，时间限制: {max_time_seconds / 60:.1f} 分钟)...")
        while iteration < max_iterations:
            if time.time() - search_start > max_time_seconds:
                print(f"\n⏰ 达到最大时间限制 {max_time_seconds} 秒，结束搜索。")
                break

            iteration += 1

            if stagnation_counter >= stagnation_limit:
                kick_jobs = np.random.choice(self.n, size=4, replace=False)
                for kj in kick_jobs:
                    current_seq = current_seq[current_seq != kj]
                    random_pos = random.randint(0, len(current_seq))
                    current_seq = np.insert(current_seq, random_pos, kj)

                current_ms = calc_makespan(current_seq, self.p)
                tabu_list.fill(0)
                stagnation_counter = 0
                continue

            candidates = np.random.choice(self.n, size=candidate_size, replace=False)

            best_neighbor_ms = 1e15
            best_neighbor_seq = None
            best_job_moved = -1

            for job in candidates:
                seq_without_job = current_seq[current_seq != job]
                best_ins_ms, best_pos = taillard_insert_eval(seq_without_job, np.int64(job), self.p)

                is_tabu = tabu_list[job] >= iteration
                aspiration = (best_ins_ms < global_best_ms)

                if (not is_tabu) or aspiration:
                    if best_ins_ms < best_neighbor_ms:
                        best_neighbor_ms = best_ins_ms
                        best_neighbor_seq = np.insert(seq_without_job, best_pos, job)
                        best_job_moved = job

            if best_neighbor_seq is None:
                stagnation_counter += 1
                continue

            delta = best_neighbor_ms - current_ms
            accept_move = False

            if delta < 0:
                accept_move = True
            else:
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    accept_move = True

            if accept_move:
                current_seq = best_neighbor_seq
                current_ms = best_neighbor_ms

                tabu_list[best_job_moved] = iteration + random.randint(25, 50)

                if current_ms < global_best_ms:
                    global_best_ms = current_ms
                    global_best_seq = current_seq.copy()
                    stagnation_counter = 0
                    print(
                        f"[{time.time() - search_start:.1f}s] 代数 {iteration} | 👑 极限突破！Makespan: {global_best_ms:.2f}")
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

        return global_best_seq, global_best_ms


# ==========================================
# 3. 执行入口
# ==========================================
# ==========================================
# 3. 执行入口与结果保存
# ==========================================
if __name__ == "__main__":
    # 初始化求解器
    ts_solver = TabuSearchPFSSP('../../data/extracted_JSSP_data.csv')

    # 将刚刚上传的 NEH 序列文件传进去作为起点！
    neh_file = '../../result/truncated_neh_taillard_.csv'
    best_sequence, best_makespan = ts_solver.optimize(initial_seq_path=neh_file, max_iterations=500000,
                                                      max_time_seconds=1800)

    print("\n🏆 【最终对比结果】")
    print(f"全局最优 Makespan: {best_makespan:.2f}")

    # --- 🌟 新增：生成最终的 CSV 调度表 ---
    # 将 numpy 数组转换为 Pandas DataFrame
    final_df = pd.DataFrame({'Job_Order': best_sequence})

    # 动态生成带有分数的文件名，例如 "final_schedule_138423.csv"
    output_filename = f"final_schedule_{int(best_makespan)}.csv"

    # 保存为 CSV 文件 (不保存行索引 index)
    final_df.to_csv(output_filename, index=False)

    print(f"📁 恭喜！最终调度排产表已成功保存至当前目录: {output_filename}")

#[1600.4s] 代数 11456 | 👑 极限突破！Makespan: 138432.16

#⏰ 达到最大时间限制 1800 秒，结束搜索。

#🏆 【最终对比结果】
#全局最优 Makespan: 138432.16
#📁 恭喜！最终调度排产表已成功保存至当前目录: final_schedule_138432.csv