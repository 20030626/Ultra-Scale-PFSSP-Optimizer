import pandas as pd
import numpy as np
import time
import random
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
# Helper: Order Crossover (OX) for Permutation
# ==========================================
def order_crossover(p1, p2):
    """
    Standard Order Crossover (OX) for permutation-based chromosomes.
    """
    size = len(p1)
    c1, c2 = [-1] * size, [-1] * size
    # Select two random crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Copy middle segment from parent to child
    c1[start:end + 1] = p1[start:end + 1]
    c2[start:end + 1] = p2[start:end + 1]

    # Fill remaining genes from the other parent in order
    def fill_offspring(child, parent):
        p_idx = (end + 1) % size
        c_idx = (end + 1) % size
        while -1 in child:
            if parent[p_idx] not in child:
                child[c_idx] = parent[p_idx]
                c_idx = (c_idx + 1) % size
            p_idx = (p_idx + 1) % size
        return child

    return fill_offspring(c1, p2), fill_offspring(c2, p1)


# ==========================================
# 3. Benchmark SOTA: Standard Genetic Algorithm
# ==========================================
def run_standard_ga(times_matrix, time_limit=1157.91, pop_size=30, mutation_rate=0.2):
    n, m = times_matrix.shape
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing Benchmark: Standard Genetic Algorithm (SGA)")
    print(f"Time Limit: {time_limit}s | Population Size: {pop_size} | Mutation Rate: {mutation_rate}")

    start_time = time.time()

    # --- Initialization: Random Population ---
    # GA typically starts with a random population to explore the entire space
    population = [list(np.random.permutation(n)) for _ in range(pop_size)]
    fitness = [calculate_makespan(ind, times_matrix) for ind in population]

    best_ms = min(fitness)
    best_seq = population[fitness.index(best_ms)].copy()
    print(f"Initial Best Makespan (Random Gen 0): {best_ms:.2f}")

    generation = 0
    improvement_count = 0

    # --- Main Evolution Loop ---
    while time.time() - start_time < time_limit:
        generation += 1
        new_population = []

        # Elitism: Keep the best individual to prevent degradation
        new_population.append(best_seq.copy())

        while len(new_population) < pop_size:
            # Tournament Selection (Size = 3)
            candidates1 = random.sample(list(zip(population, fitness)), 3)
            p1 = min(candidates1, key=lambda x: x[1])[0]

            candidates2 = random.sample(list(zip(population, fitness)), 3)
            p2 = min(candidates2, key=lambda x: x[1])[0]

            # Crossover
            c1, c2 = order_crossover(p1, p2)

            # Mutation (Swap)
            for child in (c1, c2):
                if random.random() < mutation_rate:
                    idx1, idx2 = random.sample(range(n), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]

                if len(new_population) < pop_size:
                    new_population.append(child)

        # Update population and evaluate fitness
        population = new_population
        fitness = [calculate_makespan(ind, times_matrix) for ind in population]

        current_best = min(fitness)
        if current_best < best_ms:
            best_ms = current_best
            best_seq = population[fitness.index(best_ms)].copy()
            improvement_count += 1
            print(f"[{time.time() - start_time:.1f}s] Gen {generation} | New Best Makespan: {best_ms:.2f}")

    print(f"\nSGA Benchmark Completed! Total Generations: {generation}")
    print(f"Improvements Found: {improvement_count}")
    return best_seq, best_ms


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    file_path = '../../data/extracted_JSSP_data.csv'

    try:
        times_data = load_data(file_path)

        # Give SGA exactly 1157.91 seconds for fair comparison
        best_sequence, best_makespan = run_standard_ga(times_data, time_limit=1157.91)

        print("\n" + "★" * 50)
        print(f" Benchmark: Standard Genetic Algorithm (SGA) Results")
        print(f" Final 1000x1000 Makespan: {best_makespan:.2f}")
        print("★" * 50 + "\n")

        output_filename = '../../result/sga_benchmark_sequence.csv'
        pd.DataFrame({'Job_Order': best_sequence}).to_csv(output_filename, index=False)
        print(f"✅ Baseline sequence saved to: {output_filename}")

    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}")
#Time Limit: 258s | Population Size: 30 | Mutation Rate: 0.2
#Initial Best Makespan (Random Gen 0): 146757.01
#[36.1s] Gen 1 | New Best Makespan: 146678.20
#[72.3s] Gen 3 | New Best Makespan: 146433.34
#[108.7s] Gen 5 | New Best Makespan: 146387.43

#SGA Benchmark Completed! Total Generations: 14
#Improvements Found: 3


 #Benchmark: Standard Genetic Algorithm (SGA) Results
 #Final 1000x1000 Makespan: 146387.43

#Time Limit: 600s | Population Size: 30 | Mutation Rate: 0.2
#[563.5s] Gen 31 | New Best Makespan: 145931.67

#SGA Benchmark Completed! Total Generations: 34
#Improvements Found: 8
#Benchmark: Standard Genetic Algorithm (SGA) Results
#Final 1000x1000 Makespan: 145931.67

#Time Limit: 1157.91s | Population Size: 30 | Mutation Rate: 0.2
#SGA Benchmark Completed! Total Generations: 65
#Improvements Found: 37
#Benchmark: Standard Genetic Algorithm (SGA) Results
#Final 1000x1000 Makespan: 145503.93