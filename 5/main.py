import random
import matplotlib.pyplot as plt

# Dekodowanie chromosomu
def decode(chromosome):
    coeffs = []
    for i in range(0, len(chromosome), 5):
        coeff_bits = chromosome[i:i+5]
        coeff = int(''.join(map(str, coeff_bits[1:])), 2)
        if coeff_bits[0] == 0:  # sprawdzenie znaku
            coeff *= -1
        coeffs.append(coeff)
    return coeffs

# Funkcja celu
def fitness(chromosome, data_points):
    a, b, c, d = decode(chromosome)
    return sum((a*x**3 + b*x**2 + c*x + d - y)**2 for x, y in data_points)

# Selekcja ruletki
# todo: 1 - (f/total_fitness) -> f/total_fitness
def select(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [1 - (f/total_fitness) for f in fitness_values]
    return random.choices(population, weights=selection_probs, k=len(population))

# Krzyżowanie
def crossover(ch1, ch2):
    point = random.randint(1, len(ch1) - 2)
    return ch1[:point] + ch2[point:], ch2[:point] + ch1[point:]

# Mutacja
def mutate(chromosome, mutation_rate=0.01):
    return [gene if random.random() > mutation_rate else 1-gene for gene in chromosome]

# Wizualizacja wyników
def plot_results(data_points, best_chromosome,start_best_chromosome):
    a, b, c, d = decode(best_chromosome)
    x_values = [x for x, _ in data_points]
    y_values = [y for _, y in data_points]
    plt.scatter(x_values, y_values, color='red')  # punkty danych

    x_range = sorted(x_values)
    plt.plot(x_range, [a*x**3 + b*x**2 + c*x + d for x in x_range], color='green')  # najlepsze dopasowanie

    a, b, c, d = decode(start_best_chromosome)
    plt.plot(x_range, [a*x**3 + b*x**2 + c*x + d for x in x_range], color='blue')  # najlepsze PRZED dopasowanie
    plt.show()

# Dane
data_points = [(-5, -150), (-4, -77), (-3, -30), (-2, 0), (-1, 10), (0.5, 131/8), (1, 18), (2, 25), (3, 32), (4, 75), (5, 130)]

# Parametry algorytmu
N = 10  # Rozmiar populacji
chromosome_length = 20  # 4 współczynniki * 5 bitów każdy
num_generations = 100000  # Maksymalna liczba pokoleń
mutation_rate = 0.01
stagnation_limit = 20  # Limit stagnacji

# Inicjalizacja populacji
population = [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(N)]
best_fitness = float('inf')
stagnation_counter = 0


start_best_chromosome = min(population, key=lambda ch: fitness(ch, data_points))


# Główna pętla algorytmu
for generation in range(num_generations):
    fitness_values = [fitness(ch, data_points) for ch in population]
    print(fitness_values)
    current_best_fitness = min(fitness_values)
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        stagnation_counter = 0
    else:
        stagnation_counter += 1
        if stagnation_counter >= stagnation_limit:
            break  # Zakończenie algorytmu z powodu stagnacji

    # Selekcja, krzyżowanie i mutacja
    population = select(population, fitness_values)
    new_population = []
    for i in range(0, N, 2):
        ch1, ch2 = population[i], population[i+1]
        ch1, ch2 = crossover(ch1, ch2)
        new_population.extend([mutate(ch1, mutation_rate), mutate(ch2, mutation_rate)])
    population = new_population

# Wypisanie ostatecznego najlepszego rozwiązania
best_chromosome = min(population, key=lambda ch: fitness(ch, data_points))
decoded_coeffs = decode(best_chromosome)

print(f"Najlepszy chromosom: {best_chromosome} ({decoded_coeffs}), Fitness: {fitness(best_chromosome, data_points)}")

# Wizualizacja wyników
plot_results(data_points, best_chromosome,start_best_chromosome)
