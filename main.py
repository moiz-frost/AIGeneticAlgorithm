from __future__ import division
import random, matplotlib.pyplot as plt
from random import randrange


def pair_calculate(source):
    result = []
    for p1 in range(len(source)):
        for p2 in range(p1 + 1, len(source)):
            result.append([source[p1], source[p2]])
    return result


def contains(small, big):
    for i in xrange(len(big) - len(small) + 1):
        for j in xrange(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return i, i + len(small)
    return False


def generate_random_coordinates():
    coordinates = []
    x = random.uniform(-5, +5)
    y = random.uniform(-5, +5)
    coordinates.append(x)
    coordinates.append(y)
    return coordinates


def generate_initial_population(population_size):
    population = []
    i = 0
    while i != population_size:
        new_member = generate_random_coordinates()
        if new_member not in population:
            population.append(new_member)
            i += 1
    return population


def compute_fitness_of_population(population):
    fitness_values = []
    for entity in population:
        fitness_value = (entity[0] ** 2) + (entity[1] ** 2)
        fitness_values.append(fitness_value)
    return fitness_values


def compute_fitness(x, y):
    return (x ** 2) + (y ** 2)


def mutate(probability, population):
    for elem in population:
        elem.append(compute_fitness(elem[0], elem[1]))

    for i in range(len(population)):
        if population[i][2] < 48:
            mutation_probability = randrange(0, 100, 1)
            if mutation_probability in range(0, probability, 1):
                population[i][0] += random.uniform(-0.25, 0.25)
                population[i][1] += random.uniform(-0.25, 0.25)
    return population


def crossover(population, mutation):
    non_duplicate_children_population = []
    single_dimensional_list = []
    final_required_list = []

    for element in population:
        single_dimensional_list.append(element[0])
        single_dimensional_list.append(element[1])

    possible_children_list = pair_calculate(single_dimensional_list)

    # temp_list = set(map(tuple, possible_children_list))  # Removes duplicates (iterable type)
    # possible_children_list = map(list, temp_list)  # Converts back to list type

    # Mutation chance of x%
    x = mutation

    children_with_mutation_population = mutate(5, possible_children_list)

    for child in children_with_mutation_population:
        # if child not in non_duplicate_children_population:
        non_duplicate_children_population.append(child)

    for i in range(len(population) * 4):
        final_required_list.append(non_duplicate_children_population[i])

    return final_required_list


def binary_selection(population, no_of_outputs_required):
    population_len = len(population)
    selected_candidates = []

    for i in range(no_of_outputs_required):
        rand_int_x = random.randint(0, population_len - 1)
        rand_int_y = random.randint(0, population_len - 1)
        fighter_one = population[rand_int_x]
        fighter_two = population[rand_int_y]
        while (rand_int_x == rand_int_y) or (contains(fighter_one, selected_candidates)) or (
                contains(fighter_two, selected_candidates)):
            rand_int_x = random.randint(0, population_len - 1)
            rand_int_y = random.randint(0, population_len - 1)
            fighter_one = population[random.randint(0, population_len - 1)]
            fighter_two = population[random.randint(0, population_len - 1)]
        fighter_one_fitness = compute_fitness(fighter_one[0], fighter_one[1])
        fighter_two_fitness = compute_fitness(fighter_two[0], fighter_two[1])
        if fighter_one_fitness > fighter_two_fitness:
            selected_candidates.append(fighter_one)
        else:
            selected_candidates.append(fighter_two)

    return selected_candidates


def truncation_selection_for_nextgeneraton(population, no_of_outputs_required):
    population_with_fitness_values = []
    selected_candidates = []
    for element in population:
        x = element[0]
        y = element[1]
        fitness = compute_fitness(x, y)
        population_with_fitness_values.append([x, y, fitness])

    population_with_fitness_values.sort(key=lambda element: element[2], reverse=True)  # sort by fitness

    for i in range(no_of_outputs_required):
        selected_candidates.append([population_with_fitness_values[i][0], population_with_fitness_values[i][1]])

    return selected_candidates


def roulette_wheel_selection(population, no_of_selected_outputs_required):
    probability = 0
    population_with_fitness_values_and_probabilities = []
    temp_selected_candidates = []
    selected_candidates = []
    total_fitness_sum = 0

    for element in population:
        total_fitness_sum += compute_fitness(element[0], element[1])

    for i in range(len(population)):
        x = population[i][0]
        y = population[i][1]
        fitness = compute_fitness(x, y)
        probability += fitness / total_fitness_sum
        population_with_fitness_values_and_probabilities.append([x, y, fitness, probability])

    population_with_fitness_values_and_probabilities.sort(key=lambda element: element[3])  # sort by probabilities

    while len(temp_selected_candidates) < no_of_selected_outputs_required:
        temp_random_number = random.uniform(0, 1)
        for i in range(len(population_with_fitness_values_and_probabilities)):
            if i == 0:
                if (temp_random_number <= population_with_fitness_values_and_probabilities[i][3]) and \
                        (population_with_fitness_values_and_probabilities[i] not in temp_selected_candidates):
                    temp_selected_candidates.append(population_with_fitness_values_and_probabilities[i])
                continue

            if i == len(population_with_fitness_values_and_probabilities) - 1:
                continue

            if (population_with_fitness_values_and_probabilities[i][3] < temp_random_number <=
                    population_with_fitness_values_and_probabilities[i + 1][3]) and \
                    (population_with_fitness_values_and_probabilities[i] not in temp_selected_candidates):
                temp_selected_candidates.append(population_with_fitness_values_and_probabilities[i])
                continue

    for i in range(no_of_selected_outputs_required):
        selected_candidates.append([temp_selected_candidates[i][0], temp_selected_candidates[i][1]])

    return selected_candidates


def rank_based_selection(population, no_of_selected_outputs_required):
    population_with_fitness_values = []
    selected_candidates = []
    temp_selected_candidates = []
    index = 1
    total_index_sum = 0
    probability = 0

    for element in population:
        x = element[0]
        y = element[1]
        fitness = compute_fitness(x, y)
        population_with_fitness_values.append([x, y, fitness])

    population_with_fitness_values.sort(key=lambda element: element[2])  # sort by fitness

    for elem in population_with_fitness_values:
        elem.append(index)
        index += 1

    for elem in population_with_fitness_values:
        total_index_sum += elem[3]

    for elem in population_with_fitness_values:
        index_probability = elem[3] / total_index_sum
        probability += index_probability
        elem[3] = probability

    while len(temp_selected_candidates) < no_of_selected_outputs_required:
        temp_random_number = random.uniform(0, 1)
        for i in range(len(population_with_fitness_values)):
            if i == 0:
                if (temp_random_number <= population_with_fitness_values[i][3]) and \
                        (population_with_fitness_values[i] not in temp_selected_candidates):
                    temp_selected_candidates.append(population_with_fitness_values[i])
                continue

            if i == len(population_with_fitness_values) - 1:
                continue

            if (population_with_fitness_values[i][3] < temp_random_number <=
                    population_with_fitness_values[i + 1][3]) and \
                    (population_with_fitness_values[i] not in temp_selected_candidates):
                temp_selected_candidates.append(population_with_fitness_values[i])
                continue

    for i in range(no_of_selected_outputs_required):
        selected_candidates.append([temp_selected_candidates[i][0], temp_selected_candidates[i][1]])

    return selected_candidates


def FPS_and_Truncation(generations, initial_population):
    best_fitness = []
    avg_fitness = []
    lowest_fitness = []
    iteration_values = []
    fitness_values = []
    temp_total_fitness = 0
    parent_population = initial_population

    for i in range(1, generations + 1):
        iteration_values.append(i)
        selected_parent_population = roulette_wheel_selection(parent_population, 5)
        children_population = crossover(selected_parent_population, 10)
        next_population = truncation_selection_for_nextgeneraton(children_population, 10)
        parent_population = next_population
        for entity in next_population:
            fitness = compute_fitness(entity[0], entity[1])
            temp_total_fitness += fitness
            fitness_values.append(fitness)
        fitness_values.sort(reverse=True)
        best_fitness.append(fitness_values[0])
        lowest_fitness.append(fitness_values[len(fitness_values) - 1])
        temp_avg_fitness = float(sum(fitness_values)) / len(fitness_values) if len(fitness_values) > 0 else float('nan')
        avg_fitness.append(temp_avg_fitness)

    plt.figure(1)
    plt.plot(iteration_values, avg_fitness, 'ro')
    plt.plot(iteration_values, best_fitness, 'bo')
    plt.plot(iteration_values, lowest_fitness, 'yo')
    plt.xlabel('FPS and Truncation')

    # for i in range(len(best_fitness)):
    #     print '{0} - {1} - {2} - {3}'.format(iteration_values[i], best_fitness[i], avg_fitness[i], lowest_fitness[i])


def RBS_and_Truncation(generations, initial_population):
    best_fitness = []
    avg_fitness = []
    lowest_fitness = []
    iteration_values = []
    fitness_values = []
    temp_total_fitness = 0
    parent_population = initial_population

    for i in range(1, generations + 1):
        iteration_values.append(i)
        selected_parent_population = rank_based_selection(parent_population, 5)
        children_population = crossover(selected_parent_population, 10)
        next_population = truncation_selection_for_nextgeneraton(children_population, 10)
        parent_population = next_population
        for entity in next_population:
            fitness = compute_fitness(entity[0], entity[1])
            temp_total_fitness += fitness
            fitness_values.append(fitness)
        fitness_values.sort(reverse=True)
        best_fitness.append(fitness_values[0])
        lowest_fitness.append(fitness_values[len(fitness_values) - 1])
        temp_avg_fitness = float(sum(fitness_values)) / len(fitness_values) if len(fitness_values) > 0 else float('nan')
        avg_fitness.append(temp_avg_fitness)

    plt.figure(2)
    plt.plot(iteration_values, avg_fitness, 'ro')
    plt.plot(iteration_values, best_fitness, 'bo')
    plt.plot(iteration_values, lowest_fitness, 'yo')
    plt.xlabel('RBS and Truncation')

    # for i in range(len(best_fitness)):
    #     print '{0} - {1} - {2} - {3}'.format(iteration_values[i], best_fitness[i], avg_fitness[i], lowest_fitness[i])


def BinaryTournament_and_Truncation(generations, initial_population):
    best_fitness = []
    avg_fitness = []
    lowest_fitness = []
    iteration_values = []
    fitness_values = []
    temp_total_fitness = 0
    parent_population = initial_population

    for i in range(1, generations + 1):
        iteration_values.append(i)
        selected_parent_population = binary_selection(parent_population, 5)
        children_population = crossover(selected_parent_population, 10)
        next_population = truncation_selection_for_nextgeneraton(children_population, 10)
        parent_population = next_population
        for entity in next_population:
            fitness = compute_fitness(entity[0], entity[1])
            temp_total_fitness += fitness
            fitness_values.append(fitness)
        fitness_values.sort(reverse=True)
        best_fitness.append(fitness_values[0])
        lowest_fitness.append(fitness_values[len(fitness_values) - 1])
        temp_avg_fitness = float(sum(fitness_values)) / len(fitness_values) if len(fitness_values) > 0 else float('nan')
        avg_fitness.append(temp_avg_fitness)

    plt.figure(3)
    plt.plot(iteration_values, avg_fitness, 'ro')
    plt.plot(iteration_values, best_fitness, 'bo')
    plt.plot(iteration_values, lowest_fitness, 'yo')
    plt.xlabel('Binary Tournament and Truncation')

    # for i in range(len(best_fitness)):
    #     print '{0} - {1} - {2} - {3}'.format(iteration_values[i], best_fitness[i], avg_fitness[i], lowest_fitness[i])


def FPS_BinaryTournament(generations, initial_population):
    best_fitness = []
    avg_fitness = []
    lowest_fitness = []
    iteration_values = []
    fitness_values = []
    temp_total_fitness = 0
    parent_population = initial_population

    for i in range(1, generations + 1):
        iteration_values.append(i)
        selected_parent_population = roulette_wheel_selection(parent_population, 5)
        children_population = crossover(selected_parent_population, 10)
        next_population = binary_selection(children_population, 10)
        parent_population = next_population
        for entity in next_population:
            fitness = compute_fitness(entity[0], entity[1])
            temp_total_fitness += fitness
            fitness_values.append(fitness)
        fitness_values.sort(reverse=True)
        best_fitness.append(fitness_values[0])
        lowest_fitness.append(fitness_values[len(fitness_values) - 1])
        temp_avg_fitness = float(sum(fitness_values)) / len(fitness_values) if len(fitness_values) > 0 else float('nan')
        avg_fitness.append(temp_avg_fitness)

    plt.figure(4)
    plt.plot(iteration_values, avg_fitness, 'ro')
    plt.plot(iteration_values, best_fitness, 'bo')
    plt.plot(iteration_values, lowest_fitness, 'yo')
    plt.xlabel('FPS and Binary Tournament')

    # for i in range(len(best_fitness)):
    #     print '{0} - {1} - {2} - {3}'.format(iteration_values[i], best_fitness[i], avg_fitness[i], lowest_fitness[i])


def RBS_BinaryTournament(generations, initial_population):
    best_fitness = []
    avg_fitness = []
    lowest_fitness = []
    iteration_values = []
    fitness_values = []
    temp_total_fitness = 0
    parent_population = initial_population

    for i in range(1, generations + 1):
        iteration_values.append(i)
        selected_parent_population = rank_based_selection(parent_population, 5)
        children_population = crossover(selected_parent_population, 10)
        next_population = truncation_selection_for_nextgeneraton(children_population, 10)
        parent_population = next_population
        for entity in next_population:
            fitness = compute_fitness(entity[0], entity[1])
            temp_total_fitness += fitness
            fitness_values.append(fitness)
        fitness_values.sort(reverse=True)
        best_fitness.append(fitness_values[0])
        lowest_fitness.append(fitness_values[len(fitness_values) - 1])
        temp_avg_fitness = float(sum(fitness_values)) / len(fitness_values) if len(fitness_values) > 0 else float('nan')
        avg_fitness.append(temp_avg_fitness)

    plt.figure(5)
    plt.plot(iteration_values, avg_fitness, 'ro')
    plt.plot(iteration_values, best_fitness, 'bo')
    plt.plot(iteration_values, lowest_fitness, 'yo')
    plt.xlabel('RBS and Binary Tournament')

    # for i in range(len(best_fitness)):
    #     print '{0} - {1} - {2} - {3}'.format(iteration_values[i], best_fitness[i], avg_fitness[i], lowest_fitness[i])


def BinaryTournament_BinaryTournament(generations, initial_population):
    best_fitness = []
    avg_fitness = []
    lowest_fitness = []
    iteration_values = []
    fitness_values = []
    temp_total_fitness = 0
    parent_population = initial_population

    for i in range(1, generations + 1):
        iteration_values.append(i)
        selected_parent_population = binary_selection(parent_population, 5)
        children_population = crossover(selected_parent_population, 10)
        next_population = binary_selection(children_population, 10)
        parent_population = next_population
        for entity in next_population:
            fitness = compute_fitness(entity[0], entity[1])
            temp_total_fitness += fitness
            fitness_values.append(fitness)
        fitness_values.sort(reverse=True)
        best_fitness.append(fitness_values[0])
        lowest_fitness.append(fitness_values[len(fitness_values) - 1])
        temp_avg_fitness = float(sum(fitness_values)) / len(fitness_values) if len(fitness_values) > 0 else float('nan')
        avg_fitness.append(temp_avg_fitness)

    plt.figure(6)
    plt.plot(iteration_values, avg_fitness, 'ro')
    plt.plot(iteration_values, best_fitness, 'bo')
    plt.plot(iteration_values, lowest_fitness, 'yo')
    plt.xlabel('Binary Tournament and Binary Tournament')

    # for i in range(len(best_fitness)):
    #     print '{0} - {1} - {2} - {3}'.format(iteration_values[i], best_fitness[i], avg_fitness[i], lowest_fitness[i])


initial_population = generate_initial_population(10)

FPS_and_Truncation(40, initial_population)
RBS_and_Truncation(40, initial_population)
BinaryTournament_and_Truncation(40, initial_population)
FPS_BinaryTournament(40, initial_population)
RBS_BinaryTournament(40, initial_population)
BinaryTournament_BinaryTournament(40, initial_population)

plt.show()
