import numpy as np
import random

# Data restoran dengan koordinat, rating, dan harga rata-rata
restaurants = [
    {"name": "AGP Ayam Goreng", "location": (0, 0), "rating": 4.2, "price": 100000},
    {"name": "Solaria", "location": (1, 3), "rating": 3.3, "price": 80000},
    {"name": "Astoria", "location": (2, -4), "rating": 4.6, "price": 150000},
    {"name": "The Pantry", "location": (-1, 0), "rating": 4.4, "price": 200000},
    {"name": "Jogja Steak", "location": (-1, 4), "rating": 4.2, "price": 70000},
    {"name": "Pondok Garuda", "location": (-3, 1), "rating": 4.5, "price": 120000},
    {"name": "Angkringan", "location": (-6, 1), "rating": 4.2, "price": 90000},
    {"name": "Luweh", "location": (-6, -2), "rating": 4.3, "price": 50000},
    {"name": "Nyamsir ", "location": (-8, 0), "rating": 4.5, "price": 130000},
    {"name": "Sari Alam", "location": (-7, 6), "rating": 4.3, "price": 180000},
    {"name": "Warung Podjok", "location": (-8, 3), "rating": 4.6, "price": 180000},
    {"name": "Warung Jogja", "location": (-9, 5), "rating": 4.4, "price": 180000},
]

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Fungsi untuk menghitung total jarak rute
def total_distance(route):
    distance = 0
    for i in range(len(route)):
        distance += calculate_distance(route[i]["location"], route[(i + 1) % len(route)]["location"])
    return distance

# Fungsi fitness yang mempertimbangkan jarak, rating, dan harga
def fitness(route, weight_distance=0.5, weight_rating=0.3, weight_price=0.2):
    total_dist = total_distance(route)
    total_rating = sum(r["rating"] for r in route) / len(route)
    total_price = sum(r["price"] for r in route) / len(route)
    
    # Normalisasi
    max_dist = np.sqrt((20)**2 + (20)**2) * len(route)  # Rough estimate of max distance
    norm_dist = total_dist / max_dist
    norm_rating = total_rating / 5.0  # anggap rating maksimum adalah 5
    norm_price = total_price / max(r["price"] for r in restaurants)
    
    fitness_value = (weight_distance * (1 / norm_dist)) + (weight_rating * norm_rating) - (weight_price * norm_price)
    return fitness_value

# Inisialisasi populasi
def initialize_population(pop_size, restaurants):
    population = []
    for _ in range(pop_size):
        route = restaurants[:]
        random.shuffle(route)
        population.append(route)
    return population

# Seleksi individu untuk crossover menggunakan tournament selection
def select(population, k=3):
    selected = random.sample(population, k)
    selected = sorted(selected, key=fitness, reverse=True)
    return selected[:2]

# Crossover (penyilangan)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end+1] = parent1[start:end+1]
    ptr = 0
    for loc in parent2:
        if loc not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = loc
    return child

# Mutasi
def mutate(route, mutation_rate=0.01):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]

# Algoritma Genetik
def genetic_algorithm(restaurants, pop_size=100, generations=50, mutation_rate=0.01, min_rating=4.0, max_price=150000):
    # Saring restoran berdasarkan minimal rating dan maksimal harga
    filtered_restaurants = [r for r in restaurants if r["rating"] >= min_rating and r["price"] <= max_price]
    
    if len(filtered_restaurants) == 0:
        raise ValueError("Tidak ada restoran yang memenuhi kriteria rating dan harga.")

    population = initialize_population(pop_size, filtered_restaurants)
    best_route = None
    best_fitness = float('-inf')

    for generation in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select(population)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Elitism: Ensure the best individual is carried over to the next generation
        best_individual = max(population, key=fitness)
        new_population.append(best_individual)

        population = new_population

        for route in population:
            current_fitness = fitness(route)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_route = route

        print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

    return best_route, best_fitness

# Menjalankan algoritma genetik dengan kriteria minimal rating dan maksimal harga
min_rating = 4.0
max_price = 500000
pop_size = 100
generations = 500
mutation_rate = 0.01

best_route, best_fitness = genetic_algorithm(restaurants, pop_size, generations, mutation_rate, min_rating, max_price)
print("Rute terbaik untuk rating di atas ", min_rating, " dan maksimal harga", max_price,":")
for restaurant in best_route:
    print(f"{restaurant['name']} - Rating: {restaurant['rating']}, Harga: {restaurant['price']}")
print("Best Fitness:", best_fitness)
