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
# 2. 核心调度计算函数 (仿真器)
# ==========================================
def calculate_makespan(permutation, times_matrix):
    """
    计算给定工件排列的总完工时间 (Makespan)
    """
    m = times_matrix.shape[1]
    completion_times = np.zeros(m)

    for job_idx in permutation:
        # 第一台机器的处理时间
        completion_times[0] += times_matrix[job_idx, 0]
        # 后续机器的处理时间
        for mach_idx in range(1, m):
            # 逻辑：当前工序开始时间 = max(本工件前一机完工, 上工件本机完工)
            if completion_times[mach_idx - 1] > completion_times[mach_idx]:
                completion_times[mach_idx] = completion_times[mach_idx - 1] + times_matrix[job_idx, mach_idx]
            else:
                completion_times[mach_idx] = completion_times[mach_idx] + times_matrix[job_idx, mach_idx]

    return completion_times[-1]


# ==========================================
# 3. 截断式 NEH 算法主体 (300精算 + 700追加)
# ==========================================
def run_truncated_neh(times_matrix, truncate_limit=300):
    n, m = times_matrix.shape
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 开始【截断式 NEH 算法】优化")
    print(f"总规模: {n}个工件 x {m}台机器")
    print(f"策略: 前 {truncate_limit} 个工件(大工件)精确插入，后 {n - truncate_limit} 个工件(小工件)贪心追加\n")

    start_time = time.time()

    # --- 准备阶段：按工件的总加工时间降序排列 ---
    job_sums = times_matrix.sum(axis=1)
    sorted_jobs = list(np.argsort(-job_sums))

    # --- 阶段一：核心工件的精细化 NEH 插入 (前 300 个) ---
    print(f"========== 阶段一：核心工件精细插入 (共 {truncate_limit} 个) ==========")

    # 取前两个工件，确定最佳初始顺序
    j1, j2 = sorted_jobs[0], sorted_jobs[1]
    ms1 = calculate_makespan([j1, j2], times_matrix)
    ms2 = calculate_makespan([j2, j1], times_matrix)
    current_seq = [j1, j2] if ms1 < ms2 else [j2, j1]

    # 从第 3 个工件开始，到第 300 个工件结束
    for i in range(2, truncate_limit):
        new_job = sorted_jobs[i]
        best_ms = float('inf')
        best_pos = 0

        # 尝试将新工件插入到当前序列的所有可能位置
        for pos in range(len(current_seq) + 1):
            temp_seq = current_seq[:pos] + [new_job] + current_seq[pos:]
            ms = calculate_makespan(temp_seq, times_matrix)

            if ms < best_ms:
                best_ms = ms
                best_pos = pos

        # 将新工件固定在最佳位置
        current_seq.insert(best_pos, new_job)

        # 进度打印 (每隔 10 个打印一次，让你明确看到程序在跑)
        if (i + 1) % 10 == 0 or (i + 1) == truncate_limit:
            elapsed = (time.time() - start_time) / 60
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 核心进度: {i + 1}/{truncate_limit} | 当前局部最优 Makespan: {best_ms:.2f} | 已耗时: {elapsed:.1f} 分钟")

    # --- 阶段二：尾部工件的贪心追加 (后 700 个) ---
    print(f"\n========== 阶段二：轻量工件贪心追加 (共 {n - truncate_limit} 个) ==========")
    # 直接提取后面 700 个已经按降序排好的工件
    remaining_jobs = sorted_jobs[truncate_limit:]
    # 直接像接火车一样拼在后面 (耗时 0 秒)
    final_seq = current_seq + remaining_jobs
    print("追加完成！")

    # --- 最终评估：计算 1000 个工件的最终 Makespan ---
    final_ms = calculate_makespan(final_seq, times_matrix)

    return final_seq, final_ms


# ==========================================
# 4. 执行入口
# ==========================================
if __name__ == "__main__":
    file_path = '../../data/extracted_JSSP_data.csv'

    try:
        times_data = load_data(file_path)

        start_all = time.time()

        # 调用算法，截断点设置为 300
        best_sequence, best_makespan = run_truncated_neh(times_data, truncate_limit=300)

        end_all = time.time()
        total_minutes = (end_all - start_all) / 60

        # --- 打印最终战报 ---
        print("\n" + "★" * 50)
        print(f" 计算任务圆满结束！")
        print(f" 最终 1000x1000 总完工时间 (Makespan): {best_makespan:.2f}")
        print(f" 总计运行时长: {total_minutes:.2f} 分钟")
        print("★" * 50 + "\n")

        # --- 保存结果 ---
        output_filename = 'truncated_neh_baseline_300.csv'
        pd.DataFrame({'Job_Order': best_sequence}).to_csv(output_filename, index=False)
        print(f"✅ 最优工件排序已成功保存至: {output_filename}")


    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}，请确认数据文件是否在当前目录下。")
    except Exception as e:
        print(f"❌ 运行过程中出现意外错误: {e}")

#1000x1000 总完工时间 (Makespan): 146510.16 总计运行时长: 85.89 分钟
#300截断，700追加

