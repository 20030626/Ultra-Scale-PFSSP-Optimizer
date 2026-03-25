import pandas as pd
import numpy as np
import time
import random
import math
import datetime


# ==========================================
# 1. Data Loading Module
# ==========================================
def load_data(file_path):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading data from: {file_path}...")
    df = pd.read_csv(file_path, header=None)
    return df.values


# ==========================================
# 2. Makespan Calculation (Simulation Engine)
# ==========================================
def calculate_makespan(permutation, times_matrix):
    """
    Calculate the total completion time (Makespan).
    """
    m = times_matrix.shape[1]
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
# 3. State-of-the-Art: Iterated Greedy (IG)
# ==========================================
def run_iterated_greedy(times_matrix, time_limit=258, d=4, temperature_factor=0.4):
    """
    Standard Iterated Greedy (IG) Algorithm for Flow-shop Scheduling.
    Reference: Ruiz and Stutzle (2007).
    """
    n, m = times_matrix.shape
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing Benchmark: Iterated Greedy (IG)")
    print(f"Time Limit: {time_limit} seconds | Destruction Size: {d}")

    start_time = time.time()

    # --- Initialization: Palmer's Heuristic ---
    weights = np.array([-(m - (2 * j + 1)) for j in range(m)])
    slope_indices = np.dot(times_matrix, weights)
    current_seq = list(np.argsort(-slope_indices))
    current_ms = calculate_makespan(current_seq, times_matrix)

    best_seq = current_seq.copy()
    best_ms = current_ms

    # Calculate temperature for simulated annealing acceptance criterion
    # Temp = Temperature_Factor * (Sum of processing times / (n * m * 10))
    total_p_time = np.sum(times_matrix)
    temperature = temperature_factor * (total_p_time / (n * m * 10))

    iteration = 0
    improvement_count = 0

    # --- Main IG Loop ---
    while time.time() - start_time < time_limit:
        iteration += 1

        # 1. Destruction Phase: Remove 'd' random jobs
        temp_seq = current_seq.copy()
        removed_jobs = []
        for _ in range(d):
            idx_to_remove = random.randrange(len(temp_seq))
            removed_jobs.append(temp_seq.pop(idx_to_remove))

        # 2. Construction Phase: Re-insert removed jobs greedily (NEH style)
        for job in removed_jobs:
            best_insert_ms = float('inf')
            best_insert_pos = 0

            # Evaluate all possible insertion positions
            for pos in range(len(temp_seq) + 1):
                eval_seq = temp_seq[:pos] + [job] + temp_seq[pos:]
                eval_ms = calculate_makespan(eval_seq, times_matrix)

                if eval_ms < best_insert_ms:
                    best_insert_ms = eval_ms
                    best_insert_pos = pos

            temp_seq.insert(best_insert_pos, job)

        new_ms = calculate_makespan(temp_seq, times_matrix)

        # 3. Acceptance Criterion (Simulated Annealing logic)
        if new_ms < current_ms:
            # Better solution found
            current_seq = temp_seq
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_seq = temp_seq.copy()
                improvement_count += 1
                print(f"[{time.time() - start_time:.1f}s] Global Best Improved! Makespan: {best_ms:.2f}")
        else:
            # Worse solution: accept with certain probability to escape local optima
            delta = new_ms - current_ms
            prob = math.exp(-delta / temperature)
            if random.random() <= prob:
                current_seq = temp_seq
                current_ms = new_ms

    print(f"\nIG Benchmark Completed! Total Iterations: {iteration}")
    print(f"Improvements Found: {improvement_count}")
    return best_seq, best_ms


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    file_path = '../../data/extracted_JSSP_data.csv'

    try:
        times_data = load_data(file_path)

        # 258 seconds = 4.3 minutes (to match your Hybrid algorithm)
        best_sequence, best_makespan = run_iterated_greedy(times_data, time_limit=1157.91)

        print("\n" + "★" * 50)
        print(f" Benchmark: Iterated Greedy (IG) Results")
        print(f" Final 1000x1000 Makespan: {best_makespan:.2f}")
        print("★" * 50 + "\n")

        output_filename = '../../result/ig_benchmark_sequence.csv'
        pd.DataFrame({'Job_Order': best_sequence}).to_csv(output_filename, index=False)
        print(f"✅ Baseline sequence saved to: {output_filename}")

    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}")


#[14:43:06] Executing Benchmark: Iterated Greedy (IG)
#Time Limit: 1157.91 seconds | Destruction Size: 4
#[2316.7s] Global Best Improved! Makespan: 146075.13

