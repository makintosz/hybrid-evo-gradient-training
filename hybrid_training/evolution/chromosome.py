import numpy as np

from hybrid_training.model.fully_connected import FullyConnectedModel
from hybrid_training.model.model_base import ModelBaseInterface


class Chromosome:
    def __init__(self, model_settings: dict):
        self._model_settings = model_settings

        self._model = None
        self.weights = None

    def generate(
        self, x: np.ndarray, y: np.ndarray, zero_weights_ratio: int = 0.5
    ) -> None:
        self._generate_model(x, y)
        self.weights = self._model.get_weights()
        if zero_weights_ratio > 0:
            self._zero_some_weights(zero_weights_ratio)

    def apply_weights_to_model(self) -> None:
        self._model.set_weights(self.weights)

    def mutate(self) -> None:
        random_float = np.random.uniform(0, 1)
        if random_float < self._model_settings["mutation_chromosome_probability"]:
            for array_index, array in self.weights.items():
                array["weight"] = self._mutate_array(array["weight"])

                if "bias" in array.keys():
                    array["bias"] = self._mutate_array(array["bias"])

    def _mutate_array(
        self, array: np.ndarray, avoid_zeros_mutation: bool = False
    ) -> np.ndarray:
        if array.ndim == 1:
            for i in range(len(array)):
                if array[i] == 0 and avoid_zeros_mutation:
                    continue

                random_float = np.random.uniform(0, 1)
                if random_float < self._model_settings["mutation_gene_probability"]:
                    array[i] = np.random.uniform(
                        -self._model_settings["gene_mutation_range"],
                        self._model_settings["gene_mutation_range"],
                    )

        if array.ndim == 2:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    random_float = np.random.uniform(0, 1)
                    if array[i, j] == 0 and avoid_zeros_mutation:
                        continue

                    if random_float < self._model_settings["mutation_gene_probability"]:
                        array[i, j] = np.random.uniform(
                            -self._model_settings["gene_mutation_range"],
                            self._model_settings["gene_mutation_range"],
                        )

        if array.ndim == 4:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        for l in range(array.shape[3]):
                            if array[i, j, k, l] == 0 and avoid_zeros_mutation:
                                continue

                            random_float = np.random.uniform(0, 1)
                            if (
                                random_float
                                < self._model_settings["mutation_gene_probability"]
                            ):
                                array[i, j, k, l] = np.random.uniform(
                                    -self._model_settings["gene_mutation_range"],
                                    self._model_settings["gene_mutation_range"],
                                )

        return array

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x).detach().numpy()

    def _generate_model(self, x: np.ndarray, y: np.ndarray) -> None:
        if self._model_settings["model_type"] == "fully_connected":
            self._model = FullyConnectedModel(self._model_settings, x, y)

    def get_model(self) -> ModelBaseInterface:
        return self._model

    def train_one_iteration(self) -> None:
        zero_weights_masks = self._find_zero_values()
        self._model.train_one_iteration()
        self._set_zero_values(zero_weights_masks)

    def _find_zero_values(self) -> dict:
        zero_weights = {}
        for layer_index in self.weights:
            zero_weights[layer_index] = {
                "weight": self.weights[layer_index]["weight"] == 0
            }
            if "bias" in self.weights[layer_index].keys():
                zero_weights[layer_index]["bias"] = (
                    self.weights[layer_index]["bias"] == 0
                )

        return zero_weights

    def _set_zero_values(self, zero_weights: dict) -> None:
        for layer_index in zero_weights:
            self.weights[layer_index]["weight"][zero_weights[layer_index]["weight"]] = 0
            if "bias" in zero_weights[layer_index].keys():
                self.weights[layer_index]["bias"][zero_weights[layer_index]["bias"]] = 0

    def _zero_some_weights(self, zero_weights_ratio: float) -> list[np.ndarray]:
        for array_index, array in self.weights.items():
            array["weight"] = self._zero_some_weights_in_array(
                array["weight"],
                zero_weights_ratio,
            )

            if "bias" in array.keys():
                array["bias"] = self._zero_some_weights_in_array(
                    array["bias"], zero_weights_ratio
                )

    @staticmethod
    def _zero_some_weights_in_array(array: np.ndarray, ratio: float) -> np.ndarray:
        if array.ndim == 1:
            for i in range(len(array)):
                random_float = np.random.uniform(0, 1)
                if random_float < ratio:
                    array[i] = 0

        if array.ndim == 2:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    random_float = np.random.uniform(0, 1)
                    if random_float < ratio:
                        array[i, j] = 0

        if array.ndim == 4:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        for l in range(array.shape[3]):
                            random_float = np.random.uniform(0, 1)
                            if random_float < ratio:
                                array[i, j, k, l] = 0

        return array
