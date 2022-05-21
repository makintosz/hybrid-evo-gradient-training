import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self._scale_data()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

    def _scale_data(self) -> None:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        self.x = scaler_x.fit_transform(self.x)
        self.y = scaler_y.fit_transform(self.y.reshape(-1, 1))
