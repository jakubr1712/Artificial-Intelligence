import random
import numpy as np
import matplotlib.pyplot as plt

# Decoding the chromosome
def decode(chromosome):
    coeffs = []
    for i in range(0, len(chromosome), 5):
        coeff_bits = chromosome[i:i+5]
        coeff = int(''.join(map(str, coeff_bits[1:])), 2)
        if coeff_bits[0] == 0:  # Checking the sign
            coeff *= -1
        coeffs.append(coeff)
    return coeffs

# Fitness function
def fitness(chromosome, data_points):
    a, b, c, d = decode(chromosome)
    return sum((a*x**3 + b*x**2 + c*x + d - y)**2 for x, y in data_points)
 
# Roulette wheel selection
def select(population, fitness_values):
    max_fitness = max(fitness_values)
    selection_probs = [(max_fitness - f + 1) / sum(max_fitness - f + 1 for f in fitness_values) for f in fitness_values]
    selected=random.choices(population, weights=selection_probs, k=len(population))
    return selected

# Crossover
def crossover(ch1, ch2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(ch1) - 2)
        return ch1[:point] + ch2[point:], ch2[:point] + ch1[point:]
    else:
        return ch1, ch2

# Mutation
def mutate(chromosome, mutation_rate):
    return [gene if random.random() > mutation_rate else 1-gene for gene in chromosome]

# Results visualization
def plot_results(data_points, best_chromosome, start_best_chromosome):
    x_values = [x for x, _ in data_points]
    y_values = [y for _, y in data_points]
    plt.scatter(x_values, y_values, color='red', label='Data Points')  

    x_min, x_max = min(x_values), max(x_values)
    # Using 500 points for smoothness
    x_range = np.linspace(x_min, x_max, 500)  

    # Best BEFORE optimization
    a, b, c, d = decode(start_best_chromosome)
    plt.plot(x_range, [a*x**3 + b*x**2 + c*x + d for x in x_range], color='blue', label='Before Algorithm')  
    
    # Best AFTER optimization
    a, b, c, d = decode(best_chromosome)
    plt.plot(x_range, [a*x**3 + b*x**2 + c*x + d for x in x_range], color='green', label='After Algorithm')  

    plt.legend()
    plt.show()

def run_genetic_algorithm(mutation_rate, crossover_rate, population_start):
    best_fitness = float('inf')
    stagnation_counter = 0
    population=population_start

    while True:
        fitness_values = [fitness(ch, data_points) for ch in population]
        current_best_fitness = min(fitness_values)

        # Checking if the termination condition is met
        if current_best_fitness < threshold_fitness:
            print("Found a chromosome that meets the termination criteria.")
            break

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                print("Algorithm stopped due to stagnation.")
                break

        # Selection, crossover, and mutation
        population = select(population, fitness_values)
        new_population = []
        while len(new_population) < N:
            ch1, ch2 = random.sample(population, 2)
            new_ch1, new_ch2 = crossover(ch1, ch2, crossover_rate)
            new_population.extend([mutate(new_ch1, mutation_rate), mutate(new_ch2, mutation_rate)])
        population = new_population

    best_chromosome = min(population, key=lambda ch: fitness(ch, data_points))
    return best_chromosome, fitness(best_chromosome, data_points)

# Data
data_points = [(-5, -150), (-4, -77), (-3, -30), (-2, 0), (-1, 10), (0.5, 131/8), (1, 18), (2, 25), (3, 32), (4, 75), (5, 130)]

N = 8  # Population size
chromosome_length = 20  # 4 coefficients * 5 bits each
threshold_fitness = 300  # Fitness threshold
stagnation_limit = 1000  # Stagnation limit

# Initializing the population
population_start = [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(N)]

# Initial best chromosome
start_best_chromosome = min(population_start, key=lambda ch: fitness(ch, data_points))

print(f"BEFORE: best chromosome: {start_best_chromosome} ({decode(start_best_chromosome)}), fitness: {fitness(start_best_chromosome, data_points)}")

# Experiment parameters
mutation_rates = [0.05]
crossover_rates = [1]

for mutation_rate in mutation_rates:
    for crossover_rate in crossover_rates:
        best_chromosome, best_chromosome_fitness = run_genetic_algorithm(mutation_rate, crossover_rate, population_start)
        print(f"AFTER: mutation: {mutation_rate}, crossover: {crossover_rate}, best chromosome: {best_chromosome} ({decode(best_chromosome)})  fitness: {best_chromosome_fitness}")
        plot_results(data_points, best_chromosome, start_best_chromosome)

input("Press Enter to exit the program.")
