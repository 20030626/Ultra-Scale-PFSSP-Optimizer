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
# 3. Simulated Annealing (SA)
# ==========================================
def run_simulated_annealing(times_matrix, time_limit=1157.91):
    n, m = times_matrix.shape
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing Benchmark: Simulated Annealing (SA)")
    print(f"Time Limit: {time_limit} seconds")

    start_time = time.time()

    # --- Initialization: Palmer's Heuristic (Fair starting line) ---
    weights = np.array([-(m - (2 * j + 1)) for j in range(m)])
    slope_indices = np.dot(times_matrix, weights)
    current_seq = list(np.argsort(-slope_indices))
    current_ms = calculate_makespan(current_seq, times_matrix)

    best_seq = current_seq.copy()
    best_ms = current_ms

    # --- SA Parameters ---
    initial_temp = 5000.0
    final_temp = 0.1
    # Calculate cooling rate to reach final_temp in the given time_limit roughly
    # We use a standard geometric cooling factor
    cooling_rate = 0.9995
    current_temp = initial_temp

    iteration = 0
    improvement_count = 0

    print(f"Initial Makespan (Palmer): {best_ms:.2f}")

    # --- Main SA Loop ---
    while time.time() - start_time < time_limit:
        iteration += 1
        new_seq = current_seq.copy()

        # Neighborhood structure: Swap or Insert
        idx1, idx2 = random.sample(range(n), 2)
        if random.random() < 0.5:
            new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]  # Swap
        else:
            job = new_seq.pop(idx1)
            new_seq.insert(idx2, job)  # Insert

        new_ms = calculate_makespan(new_seq, times_matrix)
        delta = new_ms - current_ms

        # Acceptance Criterion
        if delta < 0:
            current_seq = new_seq
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_seq = new_seq.copy()
                improvement_count += 1
                # Print occasionally to avoid console bottleneck
                if improvement_count % 5 == 0 or improvement_count == 1:
                    print(
                        f"[{time.time() - start_time:.1f}s] Improved! Makespan: {best_ms:.2f} | Temp: {current_temp:.2f}")
        else:
            # Accept worse solution with probability exp(-delta / Temp)
            # Avoid math overflow
            if current_temp > 0.0001:
                try:
                    prob = math.exp(-delta / current_temp)
                except OverflowError:
                    prob = 0
                if random.random() < prob:
                    current_seq = new_seq
                    current_ms = new_ms

        # Cooling schedule
        if iteration % 100 == 0:
            current_temp = max(final_temp, current_temp * cooling_rate)

    print(f"\nSA Benchmark Completed! Total Iterations: {iteration}")
    print(f"Improvements Found: {improvement_count}")
    return best_seq, best_ms


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    file_path = '../../data/extracted_JSSP_data.csv'
    try:
        times_data = load_data(file_path)
        # Give SA exactly 258 seconds (4.3 minutes)
        best_sequence, best_makespan = run_simulated_annealing(times_data, time_limit= 1157.91)

        print("\n" + "★" * 50)
        print(f" Benchmark: Simulated Annealing (SA) Results")
        print(f" Final 1000x1000 Makespan: {best_makespan:.2f}")
        print("★" * 50 + "\n")

    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}")


#258s，4.3min
#Initial Makespan (Palmer): 146367.86

#[53.9s] Improved! Makespan: 145802.87 | Temp: 5000.00

#SA Benchmark Completed! Total Iterations: 430
#Improvements Found: 16

 #Benchmark: Simulated Annealing (SA) Results
 #Final 1000x1000 Makespan: 145771.96

 #600s
#Time Limit: 600 seconds
#Initial Makespan (Palmer): 146367.86
#[1.2s] Improved! Makespan: 146363.03 | Temp: 5000.00
#[13.5s] Improved! Makespan: 146270.93 | Temp: 5000.00

#SA Benchmark Completed! Total Iterations: 1060
#Improvements Found: 9

# Benchmark: Simulated Annealing (SA) Results
# Final 1000x1000 Makespan: 146123.41
#1157.91s
#SA Benchmark Completed! Total Iterations: 1991
#Improvements Found: 21

# Benchmark: Simulated Annealing (SA) Results
 #Final 1000x1000 Makespan: 145699.10
