import pandas as pd
import numpy as np
import time
import datetime


# ==========================================
# 1. 加载数据
# ==========================================
def load_data(file_path):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 正在读取数据: {file_path}...")
    df = pd.read_csv(file_path, header=None)
    return df.values


# ==========================================
# 2. 核心调度计算函数 (仿真器 - 用于首尾评估)
# ==========================================
def calculate_makespan(permutation, times_matrix):
    """
    计算给定工件排列的总完工时间 (Makespan)
    """
    m = times_matrix.shape[1]
    if len(permutation) == 0:
        return 0

    completion_times = np.zeros(m)
    for job_idx in permutation:
        completion_times[0] += times_matrix[job_idx, 0]
        for mach_idx in range(1, m):
            if completion_times[mach_idx - 1] > completion_times[mach_idx]:
                completion_times[mach_idx] = completion_times[mach_idx - 1] + times_matrix[job_idx, mach_idx]
            else:
                completion_times[mach_idx] = completion_times[mach_idx] + times_matrix[job_idx, mach_idx]

    return completion_times[-1]


# ==========================================
# 3. Taillard 加速核心逻辑
# ==========================================
def find_best_insertion_taillard(current_seq, new_job, times_matrix):
    """
    利用 Taillard 加速原理寻找新工件的最佳插入位置
    """
    k = len(current_seq)
    m = times_matrix.shape[1]

    # 1. 计算前向矩阵 E (Earliest completion times)
    # e[i][j] 表示前 i 个工件在机器 j 上的完工时间
    e = np.zeros((k + 1, m))
    for i in range(1, k + 1):
        job_idx = current_seq[i - 1]
        e[i, 0] = e[i - 1, 0] + times_matrix[job_idx, 0]
        for j in range(1, m):
            e[i, j] = max(e[i, j - 1], e[i - 1, j]) + times_matrix[job_idx, j]

    # 2. 计算后向矩阵 Q (Relative late times / tails)
    # q[i][j] 表示从第 i 个工件到最后一个工件，在机器 j 上的流转时间
    q = np.zeros((k + 2, m))
    for i in range(k, 0, -1):
        job_idx = current_seq[i - 1]
        q[i, m - 1] = q[i + 1, m - 1] + times_matrix[job_idx, m - 1]
        for j in range(m - 2, -1, -1):
            q[i, j] = max(q[i, j + 1], q[i + 1, j]) + times_matrix[job_idx, j]

    # 3. 尝试所有可能的插入位置 pos (从 0 到 k)
    best_ms = float('inf')
    best_pos = 0

    for pos in range(k + 1):
        # 计算新工件在各个机器上的完工时间 f[j]
        f = np.zeros(m)
        f[0] = e[pos, 0] + times_matrix[new_job, 0]
        for j in range(1, m):
            f[j] = max(f[j - 1], e[pos, j]) + times_matrix[new_job, j]

        # 结合后向矩阵，计算如果插入在 pos 位置的总 Makespan
        current_max = 0
        for j in range(m):
            if f[j] + q[pos + 1, j] > current_max:
                current_max = f[j] + q[pos + 1, j]

        # 记录最优位置
        if current_max < best_ms:
            best_ms = current_max
            best_pos = pos

    return best_pos, best_ms


# ==========================================
# 4. 截断式 NEH 算法主体 (使用 Taillard 加速)
# ==========================================
def run_truncated_neh(times_matrix, truncate_limit=300):
    n, m = times_matrix.shape
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 开始【Taillard 加速 - 截断式 NEH】优化")
    print(f"总规模: {n}个工件 x {m}台机器")
    print(f"策略: 前 {truncate_limit} 个工件精确插入，后 {n - truncate_limit} 个工件贪心追加\n")

    start_time = time.time()

    # --- 准备阶段：按工件的总加工时间降序排列 ---
    job_sums = times_matrix.sum(axis=1)
    sorted_jobs = list(np.argsort(-job_sums))

    print(f"========== 阶段一：核心工件精细插入 (共 {truncate_limit} 个) ==========")

    # 取前两个工件，确定最佳初始顺序
    j1, j2 = sorted_jobs[0], sorted_jobs[1]
    ms1 = calculate_makespan([j1, j2], times_matrix)
    ms2 = calculate_makespan([j2, j1], times_matrix)
    current_seq = [j1, j2] if ms1 < ms2 else [j2, j1]

    # 从第 3 个工件开始，到 truncate_limit 结束
    for i in range(2, truncate_limit):
        new_job = sorted_jobs[i]

        # 【核心修改点】调用 Taillard 加速函数寻找最佳位置
        best_pos, best_ms = find_best_insertion_taillard(current_seq, new_job, times_matrix)

        # 将新工件固定在最佳位置
        current_seq.insert(best_pos, new_job)

        # 进度打印
        if (i + 1) % 10 == 0 or (i + 1) == truncate_limit:
            elapsed = (time.time() - start_time)
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 核心进度: {i + 1}/{truncate_limit} | 当前局部 Makespan: {best_ms:.2f} | 已耗时: {elapsed:.2f} 秒")

    # --- 阶段二：尾部工件的贪心追加 ---
    print(f"\n========== 阶段二：轻量工件贪心追加 (共 {n - truncate_limit} 个) ==========")
    remaining_jobs = sorted_jobs[truncate_limit:]
    final_seq = current_seq + remaining_jobs
    print("追加完成！")

    # --- 最终评估：计算全量工件的最终 Makespan ---
    final_ms = calculate_makespan(final_seq, times_matrix)

    return final_seq, final_ms


# ==========================================
# 5. 执行入口
# ==========================================
if __name__ == "__main__":
    file_path = '../../data/extracted_JSSP_data.csv'

    try:
        times_data = load_data(file_path)

        start_all = time.time()

        # 调用算法，截断点设置为 300
        best_sequence, best_makespan = run_truncated_neh(times_data, truncate_limit=1000)

        end_all = time.time()
        total_seconds = end_all - start_all

        print("\n" + "★" * 50)
        print(f" 计算任务圆满结束！")
        print(f" 最终总完工时间 (Makespan): {best_makespan:.2f}")
        print(f" 总计运行时长: {total_seconds:.2f} 秒")
        print("★" * 50 + "\n")

        output_filename = '../../result/truncated_neh_taillard_.csv'
        pd.DataFrame({'Job_Order': best_sequence}).to_csv(output_filename, index=False)
        print(f"✅ 最优工件排序已成功保存至: {output_filename}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}，请确认数据文件是否在当前目录下。")
    except Exception as e:
        print(f"❌ 运行过程中出现意外错误: {e}")
# 300计算任务圆满结束！轻量工件贪心追加 (共 700 个)
#最终总完工时间 (Makespan): 146510.16
# 总计运行时长: 102.65 秒

#500计算任务圆满结束！轻量工件贪心追加 (共 500 个)
 #最终总完工时间 (Makespan): 145864.83
 #总计运行时长: 293.96 秒

#1000计算任务圆满结束！
 #最终总完工时间 (Makespan): 139095.58
 #总计运行时长: 1157.91 秒