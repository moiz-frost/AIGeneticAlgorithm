import math
import random
from random import randrange

current_population = []
next_population = []


def generate_random_coordinates():
    x = random.uniform(-5, +5)
    y = random.uniform(-5, +5)
    coordinates = [x, y]
    return coordinates


def compute_fitness_of_population(population):
    fitness_values = []
    for entity in population:
        fitness_value = (entity[0] ** 2) + (entity[1] ** 2)
        fitness_values.append(fitness_value)
    return fitness_values


def compute_fitness(x, y):
    return (x ** 2) + (y ** 2)


def generate_initial_population():
    population = []
    for i in range(0, 10, 1):
        population.append(generate_random_coordinates())
    return population


def mutate(probability, children_population):
    for child in children_population:
        mutation_probability = randrange(0, 100, 1)
        if mutation_probability in range(0, probability, 1):
            child[0] += random.uniform(-0.25, 0.25)
            child[1] += random.uniform(-0.25, 0.25)
    return children_population


def crossover(population):
    count = len(population)
    crossover_pairs_done = []
    children_population = []
    for i in range(0, count / 2, 1):
        random_num = [randrange(0, count - 1, 1), randrange(0, count - 1, 1)]
        while (random_num[0] == random_num[1]) | ((random_num[0], random_num[1]) in crossover_pairs_done) | (
                    (random_num[1], random_num[0]) in crossover_pairs_done):
            random_num = [randrange(0, count, 1), randrange(0, count, 1)]
        parent_one = population[random_num[0]]
        parent_two = population[random_num[1]]
        print random_num
        temp_one_x = parent_one[0]
        temp_two_x = parent_two[0]
        parent_one[0] = parent_two[1]
        parent_one[1] = temp_two_x
        parent_two[0] = parent_one[1]
        parent_two[1] = temp_one_x
        children_population.append(parent_one)
        children_population.append(parent_two)
        crossover_pairs_done.append(random_num)

    # Mutation chance of 10%
    mutate(10, children_population)
    return children_population


def roulette_wheel_selection(population, no_of_times_to_spin):
    population.sort()
    calculated_probabilities = []
    selected_candidates = []
    total_sum = 0
    fitness_values = compute_fitness_of_population(population)
    fitness_values.sort()
    for entity in population:
        total_sum += compute_fitness(entity[0], entity[1])
    for entity in fitness_values:
        calculated_probabilities.append(entity / total_sum)

    for i in range(0, no_of_times_to_spin, 1):
        random_num = random.uniform(0, 1)
        for i in range(0, len(calculated_probabilities), 1):
            if (i == 0) & (random_num < calculated_probabilities[i]):
                print "first"
                for entity in population:
                    temp_fitness = compute_fitness(entity[0], entity[1])
                    if (temp_fitness == fitness_values[i]) & (entity not in selected_candidates):
                        selected_candidates.append(entity)
                continue

            elif (i == (len(calculated_probabilities) - 1)) & (random_num > calculated_probabilities[i]):
                print "last"
                for entity in population:
                    temp_fitness = compute_fitness(entity[0], entity[1])
                    if (temp_fitness == fitness_values[i]) & (entity not in selected_candidates):
                        selected_candidates.append(entity)
                continue

            else:
                print "normal"
                for entity in population:
                    temp_fitness = compute_fitness(entity[0], entity[1])
                    if (temp_fitness == fitness_values[i]) & (entity not in selected_candidates):
                        selected_candidates.append(entity)
                continue

    return selected_candidates


print len(roulette_wheel_selection(generate_initial_population(), 10))

