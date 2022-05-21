import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_regression

from fitness_function.mse_fitness_example import FitnessMSE
from hybrid_training.config import REPRODUCIBLE
from hybrid_training.evolution.evolve import evolve

if REPRODUCIBLE:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

architecture = {
    0: {"type": "fc", "neurons_in": 12, "neurons_out": 16, "bias": True},
    1: {"type": "relu"},
    2: {"type": "fc", "neurons_in": 16, "neurons_out": 8, "bias": True},
    3: {"type": "relu"},
    4: {"type": "fc", "neurons_in": 8, "neurons_out": 4, "bias": True},
    5: {"type": "relu"},
    6: {"type": "fc", "neurons_in": 4, "neurons_out": 1, "bias": True},
    7: {"type": "sigmoid"},
}
model_settings = {
    "model_type": "fully_connected",
    "architecture": architecture,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "generations_number": 2500,
    "population_size": 30,
    "input_features": 12,
    "mutation_chromosome_probability": 0.2,
    "mutation_gene_probability": 0.01,
    "gene_mutation_range": 0.1,
    "crossover_chromosome_probability": 0.25,
    "crossover_gene_probability": 0.5,
    "device": "cpu",
}

x, y = make_regression(n_samples=1000, n_features=12)
fitness_function = FitnessMSE(x, y)
history = evolve(x, y, model_settings, fitness_function)

plt.plot(history["rmse"])
plt.show()
