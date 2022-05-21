import numpy as np

from hybrid_training.evolution.population import Population
from hybrid_training.fitness_base import FitnessFunctionBase


def evolve(
    x: np.ndarray, y: np.ndarray, settings: dict, fitness: FitnessFunctionBase
) -> dict:
    population = Population(settings)
    population.generate(x, y)
    for current_generation in range(settings["generations_number"]):
        print(current_generation)
        population.calculate_fitness(fitness)
        population.select_chromosomes()
        population.mutate_population()
        population.crossover_population()
        population.conduct_gradient_step()
        population.apply_new_generation()

    return fitness.get_metrics()
