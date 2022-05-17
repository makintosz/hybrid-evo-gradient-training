from hybrid_training.evolution.population import Population


architecture = {
    0: {"type": "fc", "neurons_in": 3, "neurons_out": 16, "bias": True},
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
    "population_size": 20,
    "mutation_chromosome_probability": 0.3,
    "mutation_gene_probability": 0.2,
    "gene_mutation_range": 0.1,
    "crossover_chromosome_probability": 0.25,
    "crossover_gene_probability": 0.5,
}

pop = Population(model_settings)
pop.generate()
pop.mutate_population()
pop.crossover_population()
