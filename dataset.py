from typing import Tuple
import numpy as np

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    """
    Dataset de torch que contiene las secuencias de notas y los targets (notas a predecir).
    """
    def __init__(self, X, y, label_encoder):
        self.X = X # (num_samples, seq_length, num_features) 
        self.y = y # (num_samples, num_features)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Devuelve una secuencia de notas (X) y la nota objetivo (y), lista para el modelo.

        Salida:
        - X: (x_pitch, x_step, x_duration, x_velocity)
        - Y: (y_pitch, y_step, y_duration, y_velocity)
        """

        x_seq = self.X[idx] 
        y_target = self.y[idx]

        # inputs (X)
        x_pitch = torch.tensor(self.label_encoder.transform(x_seq[:, 0]), dtype=torch.long).unsqueeze(1)
        x_step = torch.tensor(x_seq[:, 1].astype(float), dtype=torch.float).unsqueeze(1)
        x_duration = torch.tensor(x_seq[:, 2].astype(float), dtype=torch.float).unsqueeze(1)
        x_velocity = torch.tensor(x_seq[:, 3].astype(float), dtype=torch.float).unsqueeze(1)    

        # y
        y_pitch = torch.tensor(self.label_encoder.transform([y_target[0]])[0], dtype=torch.long)
        y_step = torch.tensor(float(y_target[1]), dtype=torch.float)
        y_duration = torch.tensor(float(y_target[2]), dtype=torch.float)
        y_velocity = torch.tensor(float(y_target[3]), dtype=torch.float)

        return (x_pitch, x_step, x_duration, x_velocity), (y_pitch, y_step, y_duration, y_velocity)
