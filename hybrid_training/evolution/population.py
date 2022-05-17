from copy import deepcopy

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from hybrid_training.evolution.chromosome import Chromosome
from hybrid_training.fitness_base import FitnessFunctionBase


class Population:
    def __init__(self, settings: dict) -> None:
        self._settings = settings

        self._chromosomes = []
        self._chromosomes_new = []
        self._fitness = []

    def generate(self) -> None:
        for chromosome_index in range(self._settings["population_size"]):
            chromosome = Chromosome(self._settings)
            chromosome.generate()
            self._chromosomes.append(chromosome)

    def calculate_fitness(self, fitness: FitnessFunctionBase) -> None:
        self._fitness = fitness.calculate_for_population(self._chromosomes)

    def select_chromosomes(self) -> None:
        fitness_scaler = MinMaxScaler()
        fitness_population = np.array(self._fitness)
        fitness_population = (
            fitness_scaler.fit_transform(fitness_population.reshape((-1, 1)))
            .flatten()
            .tolist()
        )
        sum_fitness_population = sum(fitness_population)
        probability_population = []
        for fitness in fitness_population:
            probability_population.append(fitness / sum_fitness_population)

        distribution_population = [0.0]
        distribution_temp = 0.0
        for prob in probability_population:
            distribution_temp += prob
            distribution_population.append(distribution_temp)

        distribution_population[-1] = 1.0
        for j in range(self._settings["population_size"]):
            random_prob = np.random.uniform(0, 1)
            for k in range(self._settings["population_size"]):
                if (
                    distribution_population[k]
                    <= random_prob
                    <= distribution_population[k + 1]
                ):
                    self._chromosomes_new.append(deepcopy(self._chromosomes[k]))

    def mutate_population(self) -> None:
        for chromosome in self._chromosomes_new:
            chromosome.mutate()

    def crossover_population(self) -> None:
        to_cross = []
        for c in range(self._settings["population_size"]):
            random_prob = np.random.uniform(0, 1)
            if random_prob < self._settings["crossover_chromosome_probability"]:
                to_cross.append(c)

        if self._is_even(to_cross) == 0:
            del to_cross[-1]

        np.random.shuffle(to_cross)
        print(len(to_cross))
        for pdx in range(0, len(to_cross), 2):
            print(pdx)
            (
                self._chromosomes[to_cross[pdx]],
                self._chromosomes[to_cross[pdx + 1]],
            ) = self._chromosomes_crossover(
                self._chromosomes[to_cross[pdx]],
                self._chromosomes[to_cross[pdx + 1]],
            )

    def _chromosomes_crossover(
        self,
        chromosome1: Chromosome,
        chromosome2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        for layer_index, layer in chromosome1.weights.items():
            (
                chromosome1.weights[layer_index]["weight"],
                chromosome2.weights[layer_index]["weight"],
            ) = self._cross(
                chromosome1.weights[layer_index]["weight"],
                chromosome2.weights[layer_index]["weight"],
            )
            if "bias" in layer.keys():
                (
                    chromosome1.weights[layer_index]["bias"],
                    chromosome2.weights[layer_index]["bias"],
                ) = self._cross(
                    chromosome1.weights[layer_index]["bias"],
                    chromosome2.weights[layer_index]["bias"],
                )

        return chromosome1, chromosome2

    def _cross(
        self, array1: np.ndarray, array2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if array1.ndim == 1:
            for i in range(len(array1)):
                if (
                    np.random.uniform(0, 1)
                    < self._settings["crossover_gene_probability"]
                ):
                    weight_temp_c1 = array1[i]
                    weight_temp_c2 = array2[i]
                    array1[i] = weight_temp_c2
                    array2[i] = weight_temp_c1

        if array1.ndim == 2:
            for i in range(array1.shape[0]):
                for j in range(array1.shape[1]):
                    if (
                        np.random.uniform(0, 1)
                        < self._settings["crossover_gene_probability"]
                    ):
                        weight_temp_c1 = array1[i, j]
                        weight_temp_c2 = array2[i, j]
                        array1[i, j] = weight_temp_c2
                        array2[i, j] = weight_temp_c1

        if array1.ndim == 4:
            for i in range(array1.shape[0]):
                for j in range(array1.shape[1]):
                    for k in range(array1.shape[2]):
                        for l in range(array1.shape[3]):
                            if (
                                np.random.uniform(0, 1)
                                < self._settings["crossover_gene_probability"]
                            ):
                                weight_temp_c1 = array1[i, j, k, l]
                                weight_temp_c2 = array2[i, j, k, l]
                                array1[i, j, k, l] = weight_temp_c2
                                array2[i, j, k, l] = weight_temp_c1

        return array1, array2

    @staticmethod
    def _is_even(seq: list) -> int:
        if len(seq) % 2 == 0:
            return 1
        else:
            return 0
