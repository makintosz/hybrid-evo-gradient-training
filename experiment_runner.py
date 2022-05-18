from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from hybrid_training.evolution.evolve import evolve
from fitness_function.mse_fitness_example import FitnessMSE


architecture = {
    0: {"type": "fc", "neurons_in": 12, "neurons_out": 16, "bias": True},
    1: {"type": "relu"},
    2: {"type": "bn"},
    3: {"type": "fc", "neurons_in": 16, "neurons_out": 8, "bias": True},
    4: {"type": "relu"},
    5: {"type": "bn"},
    6: {"type": "fc", "neurons_in": 8, "neurons_out": 4, "bias": True},
    7: {"type": "relu"},
    8: {"type": "bn"},
    9: {"type": "fc", "neurons_in": 4, "neurons_out": 1, "bias": True},
    10: {"type": "sigmoid"},
}
model_settings = {
    "model_type": "fully_connected",
    "architecture": architecture,
    "generations_number": 1500,
    "population_size": 50,
    "input_features": 12,
    "mutation_chromosome_probability": 0.2,
    "mutation_gene_probability": 0.01,
    "gene_mutation_range": 0.1,
    "crossover_chromosome_probability": 0.25,
    "crossover_gene_probability": 0.5,
}

x, y = make_regression(n_samples=1000, n_features=12)

fitness_function = FitnessMSE(x, y)

history = evolve(model_settings, fitness_function)
print(history)
plt.plot(history["rmse"])
plt.show()
