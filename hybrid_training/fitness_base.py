from abc import ABC, abstractmethod

from hybrid_training.evolution.chromosome import Chromosome


class FitnessFunctionBase(ABC):
    @abstractmethod
    def calculate_for_population(self, chromosomes: list[Chromosome]) -> list[float]:
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        pass
