import os

import torch.nn as nn
import torch
import numpy as np

from hybrid_training.model.model_base import ModelBaseInterface


class FullyConnectedModel(ModelBaseInterface):
    def __init__(self, settings: dict) -> None:
        self.settings = settings
        self.network = FullyConnectedArchitecture(settings)

    def predict(self, x: np.ndarray) -> int:
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return self.network(x.view(-1, 1, self.settings[0]["neurons_in"]))

    def get_weights(self) -> dict[int, dict[str, np.ndarray]]:
        layers_weights = {}
        for layer_index in range(len(self.settings)):
            layer = self.settings[layer_index]
            if layer["type"] == "fc":
                if layer["bias"]:
                    layer_weight = {
                        "weight": self.network.layers[layer_index].weight.data.numpy(),
                        "bias": self.network.layers[layer_index].bias.data.numpy(),
                    }

                else:
                    layer_weight = {
                        "weight": self.network.layers[layer_index].weight.data.numpy()
                    }

                layers_weights[layer_index] = layer_weight

        return layers_weights

    def set_weights(self, weights: dict[int, dict[str, np.ndarray]]) -> None:
        for layer_index, layer in weights.items():
            self.network.layers[layer_index].weight.data = torch.from_numpy(
                layer["weight"]
            )
            if "bias" in layer.keys():
                self.network.layers[layer_index].bias.data = torch.from_numpy(
                    layer["bias"]
                )

    def save_model(self) -> None:
        torch.save(self.network, os.path.join("results", "fc_model.pt"))

    def load_model(self) -> None:
        self.network = torch.load(os.path.join("results", "fc_model.pt"))
        self.network.eval()


class FullyConnectedArchitecture(nn.Module):
    def __init__(self, settings: dict) -> None:
        super().__init__()

        self.layers = nn.Sequential(*self._create_architecture(settings))

    def forward(self, x):
        x = self.layers(x)
        return x

    @staticmethod
    def _create_architecture(architecture: dict) -> list:
        layers = []
        for layer_index in range(len(architecture)):
            layer = architecture[layer_index]
            if layer["type"] == "fc":
                layers.append(
                    nn.Linear(
                        layer["neurons_in"], layer["neurons_out"], bias=layer["bias"]
                    ),
                )

            if layer["type"] == "relu":
                layers.append(nn.ReLU())

            if layer["type"] == "bn":
                layers.append(nn.BatchNorm1d(1))

            if layer["type"] == "sigmoid":
                layers.append(nn.Sigmoid())

        return layers
