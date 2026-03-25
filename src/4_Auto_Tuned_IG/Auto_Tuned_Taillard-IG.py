import pandas as pd
import numpy as np
import time
import random
import math
import datetime


# ==========================================
# 1. 数据加载与仿真器
# ==========================================
def load_data(file_path):
    return pd.read_csv(file_path, header=None).values


def load_initial_sequence(file_path):
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
# 2. Taillard 加速引擎 (O(m) 极速评估)
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
# 3. 核心评估算子 (单次参数运行)
# ==========================================
def run_single_config(times_matrix, initial_seq, time_limit, d, temp_factor, is_phase2=False):
    n, m = times_matrix.shape
    current_seq = initial_seq.copy()
    current_ms = calculate_makespan(current_seq, times_matrix)
    best_seq = current_seq.copy()
    best_ms = current_ms

    if temp_factor > 0:
        total_p_time = np.sum(times_matrix)
        temperature = temp_factor * (total_p_time / (n * m * 10))
    else:
        temperature = 0.0

    start_time = time.time()
    iteration = 0
    sa_accepts = 0

    while time.time() - start_time < time_limit:
        iteration += 1
        temp_seq = current_seq.copy()

        # 破坏阶段
        removed_jobs = []
        for _ in range(d):
            idx_to_remove = random.randrange(len(temp_seq))
            removed_jobs.append(temp_seq.pop(idx_to_remove))

        # Taillard 重建阶段
        new_ms = 0
        for job in removed_jobs:
            best_pos, new_ms = find_best_insertion_taillard(temp_seq, job, times_matrix)
            temp_seq.insert(best_pos, job)

        # 接受准则 (包含模拟退火)
        if new_ms < current_ms:
            current_seq = temp_seq.copy()
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_seq = temp_seq.copy()
                # 只有在 Phase 2 的长跑中才打印破纪录日志，避免 Phase 1 刷屏
                if is_phase2:
                    print(f"    [+{(time.time() - start_time) / 60:.1f}分钟] 破纪录! 新 Makespan: {best_ms:.2f}")
        else:
            if temperature > 0:
                delta = new_ms - current_ms
                prob = math.exp(-delta / temperature)
                if random.random() <= prob:
                    current_seq = temp_seq.copy()
                    current_ms = new_ms
                    sa_accepts += 1

    return best_seq, best_ms, iteration, sa_accepts


# ==========================================
# 4. 全自动流水线主脑 (Pipeline Controller)
# ==========================================
if __name__ == "__main__":
    matrix_file = '../../data/extracted_JSSP_data.csv'
    # 这里填你目前掌握的最优起点，比如跑出 139041 的序列，如果没有就用 139095 的文件
    sequence_file = '../../result/truncated_neh_taillard_.csv'

    # ------------------ 引擎配置区 ------------------
    phase1_time = 60  # 海选单组测试时间 (秒)
    phase2_time = 7200  # 决赛单组长跑时间 (秒)。7200秒 = 2小时
    top_k = 3  # 从海选中晋级决赛的参数数量
    # ------------------------------------------------

    try:
        times_data = load_data(matrix_file)
        init_sequence = load_initial_sequence(sequence_file)
        init_ms = calculate_makespan(init_sequence, times_data)

        print("\n" + "★" * 60)
        print(" 运筹学端到端自动调优框架 (Auto-SOTA Pipeline)")
        print(f" 起点 Makespan: {init_ms:.2f}")
        print("★" * 60 + "\n")

        # ======================================================
        # Phase 1: 自动参数海选 (Grid Search)
        # ======================================================
        print(">>> 正在启动 Phase 1: 自动参数海选 (快速探测) <<<")
        d_values = [1, 2, 3, 4]
        temp_factors = [0.0, 2.0, 5.0, 10.0]

        phase1_results = []
        total_runs = len(d_values) * len(temp_factors)
        current_run = 0

        for d in d_values:
            for tf in temp_factors:
                current_run += 1
                print(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 测试 ({current_run}/{total_runs}) d={d}, TempFactor={tf} ...",
                    end="", flush=True)

                _, best_ms, iters, sa_acc = run_single_config(times_data, init_sequence, time_limit=phase1_time, d=d,
                                                              temp_factor=tf, is_phase2=False)

                print(f" 最终: {best_ms:.2f}")
                phase1_results.append({'d': d, 'TempFactor': tf, 'Makespan': round(best_ms, 2)})

        # -------- 自动提取 Top K --------
        df_results = pd.DataFrame(phase1_results)
        # 按照 Makespan 从小到大排序
        df_results.sort_values(by='Makespan', ascending=True, inplace=True)
        # 截取前 top_k 名，转换为字典列表
        top_configs = df_results.head(top_k).to_dict('records')

        print("\n" + "=" * 50)
        print(f"🏆 Phase 1 结束！自动提取的 Top-{top_k} 晋级参数：")
        for i, config in enumerate(top_configs):
            print(
                f"  第 {i + 1} 名: d={config['d']}, TempFactor={config['TempFactor']} (短测成绩: {config['Makespan']})")
        print("=" * 50 + "\n")

        # ======================================================
        # Phase 2: 自动精英长跑 (Deep Exploitation)
        # ======================================================
        print(f">>> 正在启动 Phase 2: 精英彻夜长跑 (每组 {phase2_time / 3600:.1f} 小时) <<<")
        print(f"总预计耗时: {top_k * phase2_time / 3600:.1f} 小时。您可以去休息了！\n")

        global_best_ms = init_ms
        global_best_seq = init_sequence.copy()
        best_config_name = "Initial"

        for i, config in enumerate(top_configs):
            d = config['d']
            tf = config['TempFactor']
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚀 启动决赛组别 {i + 1}/{top_k} (d={d}, TempFactor={tf})")

            # 进入深搜模式，开启实时破纪录打印
            final_seq, final_ms, iters, acc = run_single_config(times_data, init_sequence, time_limit=phase2_time, d=d,
                                                                temp_factor=tf, is_phase2=True)

            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🏁 组别 {i + 1} 完赛! 本组最好成绩: {final_ms:.2f}")

            # 刷新全局历史最佳，并安全落袋
            if final_ms < global_best_ms:
                global_best_ms = final_ms
                global_best_seq = final_seq.copy()
                best_config_name = f"d={d}_Temp={tf}"

                safe_filename = f'AUTO_SAFE_SAVE_{int(global_best_ms)}.csv'
                pd.DataFrame({'Job_Order': global_best_seq}).to_csv(safe_filename, index=False)
                print(f"  🛡️ [自动保护] 全局新 SOTA 诞生！已暂存至: {safe_filename}\n")
            else:
                print("  (本组未能打破全局纪录)\n")

        # ======================================================
        # 最终汇报
        # ======================================================
        print("\n" + "👑" * 30)
        print(f" 自动化流水线全部执行完毕！")
        print(f" 击穿下界的王牌参数: {best_config_name}")
        print(f" 最终极光 Makespan: {global_best_ms:.2f}")
        print("👑" * 30 + "\n")

        output_filename = f'AUTO_NIGHT_SOTA_{int(global_best_ms)}.csv'
        pd.DataFrame({'Job_Order': global_best_seq}).to_csv(output_filename, index=False)
        print(f"✅ 全局最优序列已成功写入: {output_filename}")

    except Exception as e:
        print(f"❌ 运行报错: {e}")

#[+90.1分钟] 破纪录! 新 Makespan: 138921.39
#[08:37:30] 🏁 组别 3 完赛! 本组最好成绩: 138921.39
#[自动保护] 全局新 SOTA 诞生！已暂存至: AUTO_SAFE_SAVE_138921.csv


 #自动化流水线全部执行完毕！
 #击穿下界的王牌参数: d=1_Temp=2.0
 #最终极光 Makespan: 138921.39

#全局最优序列已成功写入: AUTO_NIGHT_SOTA_138921.csv