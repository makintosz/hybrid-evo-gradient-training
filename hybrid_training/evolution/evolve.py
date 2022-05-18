from hybrid_training.evolution.population import Population
from hybrid_training.fitness_base import FitnessFunctionBase


def evolve(settings: dict, fitness: FitnessFunctionBase) -> dict:
    population = Population(settings)
    population.generate()
    for current_generation in range(settings["generations_number"]):
        print(current_generation)
        population.calculate_fitness(fitness)
        population.select_chromosomes()
        population.mutate_population()
        population.crossover_population()
        population.apply_new_generation()

    return fitness.get_metrics()
