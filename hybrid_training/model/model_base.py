from abc import ABC, abstractmethod

import numpy as np


class ModelBaseInterface(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_weights(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def set_weights(self, weights: list[np.ndarray]) -> None:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def train_one_iteration(self) -> None:
        pass
