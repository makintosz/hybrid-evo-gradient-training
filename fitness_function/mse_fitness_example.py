import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from hybrid_training.fitness_base import FitnessFunctionBase
from hybrid_training.evolution.chromosome import Chromosome


class FitnessMSE(FitnessFunctionBase):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = x
        self._y = y
        self._x_unscaled = x
        self._y_unscaled = y
        self._scaler_x = None
        self._scaler_y = None
        # TODO develop metrics system
        self._metrics = {"rmse": []}
        self._scale_data()

    def calculate_for_population(self, chromosomes: list[Chromosome]) -> list[float]:
        population_scores = []
        population_metrics = {"rmse": []}
        for chromosome in chromosomes:
            chromosome.apply_weights_to_model()
            y_pred = chromosome.predict(self._x)
            mse = mean_squared_error(self._y.reshape(-1, 1), y_pred.reshape(-1, 1))
            population_scores.append(mse)
            y_pred_unscaled = self._unscale_prediction(y_pred)
            rmse = np.sqrt(
                mean_squared_error(
                    self._y_unscaled.reshape(-1, 1), y_pred_unscaled.reshape(-1, 1)
                )
            )
            population_metrics["rmse"].append(rmse)

        self._metrics["rmse"].append(np.min(population_metrics["rmse"]))
        return population_scores

    def _scale_data(self) -> None:
        self._scaler_x = MinMaxScaler()
        self._scaler_y = MinMaxScaler()
        self._x = self._scaler_x.fit_transform(self._x)
        self._y = self._scaler_y.fit_transform(self._y.reshape(-1, 1))

    def _unscale_prediction(self, y: np.ndarray) -> np.ndarray:
        return self._scaler_y.inverse_transform(y.reshape(-1, 1))

    def get_metrics(self) -> dict:
        return self._metrics
