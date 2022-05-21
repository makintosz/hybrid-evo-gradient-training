import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hybrid_training.data_processing.datasets import TabularDataset
from hybrid_training.model.model_base import ModelBaseInterface

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class FullyConnectedModel(ModelBaseInterface):
    def __init__(self, settings: dict, x: np.ndarray, y: np.ndarray) -> None:
        self._settings = settings
        self._architecture = settings["architecture"]
        self._network = FullyConnectedArchitecture(self._architecture)
        self._dataloader = DataLoader(
            dataset=TabularDataset(x, y),
            batch_size=self._settings["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        self._optimizer = optim.Adam(
            self._network.parameters(), lr=self._settings["learning_rate"]
        )
        self._loss = nn.MSELoss()

    def predict(self, x: np.ndarray) -> int:
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        return self._network(x.view(-1, 1, self._architecture[0]["neurons_in"]))

    def get_weights(self) -> dict[int, dict[str, np.ndarray]]:
        layers_weights = {}
        for layer_index in range(len(self._architecture)):
            layer = self._architecture[layer_index]
            if layer["type"] == "fc":
                if layer["bias"]:
                    layer_weight = {
                        "weight": self._network.layers[layer_index].weight.data.numpy(),
                        "bias": self._network.layers[layer_index].bias.data.numpy(),
                    }

                else:
                    layer_weight = {
                        "weight": self._network.layers[layer_index].weight.data.numpy()
                    }

                layers_weights[layer_index] = layer_weight

        return layers_weights

    def set_weights(self, weights: dict[int, dict[str, np.ndarray]]) -> None:
        for layer_index, layer in weights.items():
            self._network.layers[layer_index].weight.data = torch.from_numpy(
                layer["weight"]
            )
            if "bias" in layer.keys():
                self._network.layers[layer_index].bias.data = torch.from_numpy(
                    layer["bias"]
                )

    def train_one_iteration(self) -> None:
        for x_batch, y_batch in self._dataloader:
            self._optimizer.zero_grad()
            output = self._network(
                x_batch.view(-1, self._architecture[0]["neurons_in"])
            )
            loss = self._loss(output.view(-1, 1), y_batch)
            loss.backward()
            self._optimizer.step()

    def save_model(self) -> None:
        torch.save(self._network, os.path.join("results", "fc_model.pt"))

    def load_model(self) -> None:
        self._network = torch.load(os.path.join("results", "fc_model.pt"))
        self._network.eval()


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

            if layer["type"] == "sigmoid":
                layers.append(nn.Sigmoid())

        return layers
