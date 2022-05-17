import numpy as np

from hybrid_training.model.model_base import ModelBaseInterface
from hybrid_training.model.fully_connected import FullyConnectedModel


class Chromosome:
    def __init__(self, model_settings: dict):
        self._model_settings = model_settings

        self._model = None
        self.weights = None

    def generate(self, zero_weights_ratio: int = 0.5) -> None:
        self._generate_model()
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

    def predict(self, x: np.ndarray) -> int:
        return self._model.predict(x)

    def _generate_model(self) -> None:
        if self._model_settings["model_type"] == "fully_connected":
            self._model = FullyConnectedModel(self._model_settings["architecture"])

    def get_model(self) -> ModelBaseInterface:
        return self._model

    def _zero_some_weights(self, zero_weights_ratio: float) -> list[np.ndarray]:
        for array_index, array in self.weights.items():
            array["weight"] = self._zero_some_weights_in_array(
                array["weight"], zero_weights_ratio,
            )

            if "bias" in array.keys():
                array["bias"] = self._zero_some_weights_in_array(
                    array["bias"], zero_weights_ratio
                )

    @staticmethod
    def _zero_some_weights_in_array(
        array: np.ndarray, ratio: float
    ) -> np.ndarray:
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
