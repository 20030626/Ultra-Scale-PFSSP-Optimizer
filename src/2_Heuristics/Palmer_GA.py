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
# Helper: Palmer's Algorithm for Seeding
# ==========================================
def get_palmer_sequence(times_matrix):
    """
    Generates a high-quality initial sequence using Palmer's Slope Index.
    """
    n, m = times_matrix.shape
    weights = np.array([2 * j - m - 1 for j in range(1, m + 1)])
    slope_indices = np.dot(times_matrix, weights)
    return list(np.argsort(-slope_indices))


# ==========================================
# Helper: Order Crossover (OX)
# ==========================================
def order_crossover(p1, p2):
    size = len(p1)
    c1, c2 = [-1] * size, [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    c1[start:end + 1] = p1[start:end + 1]
    c2[start:end + 1] = p2[start:end + 1]

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
# 3. Hybrid Genetic Algorithm (Palmer + GA)
# ==========================================
def run_palmer_seeded_ga(times_matrix, time_limit=1157.91, pop_size=30, mutation_rate=0.2):
    n, m = times_matrix.shape
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing: Palmer-Seeded GA")
    print(f"Time Limit: {time_limit}s | Pop Size: {pop_size}")

    start_time = time.time()

    # --- Initialization with Palmer Seed ---
    population = []
    # Seed 1: The Palmer sequence
    palmer_seq = get_palmer_sequence(times_matrix)
    population.append(palmer_seq)

    # Fill the rest with random permutations
    for _ in range(pop_size - 1):
        population.append(list(np.random.permutation(n)))

    fitness = [calculate_makespan(ind, times_matrix) for ind in population]

    best_ms = min(fitness)
    best_seq = population[fitness.index(best_ms)].copy()

    # Let the user know the quality of the seed
    palmer_ms = fitness[0]
    print(f"Palmer Seed Makespan: {palmer_ms:.2f}")
    print(f"Initial Best Makespan (Gen 0): {best_ms:.2f}")

    generation = 0
    improvement_count = 0

    # --- Evolution Loop ---
    while time.time() - start_time < time_limit:
        generation += 1
        new_population = []
        new_population.append(best_seq.copy())  # Elitism

        while len(new_population) < pop_size:
            # Tournament Selection
            p1 = min(random.sample(list(zip(population, fitness)), 3), key=lambda x: x[1])[0]
            p2 = min(random.sample(list(zip(population, fitness)), 3), key=lambda x: x[1])[0]

            # Crossover
            c1, c2 = order_crossover(p1, p2)

            # Mutation
            for child in (c1, c2):
                if random.random() < mutation_rate:
                    idx1, idx2 = random.sample(range(n), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
                if len(new_population) < pop_size:
                    new_population.append(child)

        population = new_population
        fitness = [calculate_makespan(ind, times_matrix) for ind in population]

        current_best = min(fitness)
        if current_best < best_ms:
            best_ms = current_best
            best_seq = population[fitness.index(best_ms)].copy()
            improvement_count += 1
            print(f"[{time.time() - start_time:.1f}s] Gen {generation} | New Best Makespan: {best_ms:.2f}")

    return best_seq, best_ms


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    file_path = '../../data/extracted_JSSP_data.csv'
    try:
        times_data = load_data(file_path)
        # Testing with 1157.91s
        best_sequence, best_makespan = run_palmer_seeded_ga(times_data, time_limit=1157.91)

        print("\n" + "★" * 50)
        print(f" Final Results (Palmer-Seeded GA)")
        print(f" Final 1000x1000 Makespan: {best_makespan:.2f}")
        print("★" * 50 + "\n")

    except FileNotFoundError:
        print(f"❌ Error: File not found.")
##Time Limit: 1157.91s | Population Size: 30 | Mutation Rate: 0.2
#[1104.0s] Gen 60 | New Best Makespan: 144668.63
