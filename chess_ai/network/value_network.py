import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        super(ValueNetwork, self).__init__()
        input_size = 12 * 8 * 8
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

    def evaluate_position(self, board_tensor: np.ndarray, device: torch.device = None) -> float:
        if device is None:
            device = next(self.parameters()).device
        x = torch.from_numpy(board_tensor).unsqueeze(0).float().to(device)
        with torch.no_grad():
            value = self.forward(x).squeeze().item()
        return value

    def evaluate_positions_batch(self, board_tensors: np.ndarray, device: torch.device = None) -> np.ndarray:
        if device is None:
            device = next(self.parameters()).device
        x = torch.from_numpy(board_tensors).float().to(device)
        with torch.no_grad():
            values = self.forward(x).squeeze().cpu().numpy()
        return values


class ImprovedValueNetwork(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super(ImprovedValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.tanh(self.fc3(x))
        return x

    def evaluate_position(self, board_tensor: np.ndarray, device: torch.device = None) -> float:
        if device is None:
            device = next(self.parameters()).device
        x = torch.from_numpy(board_tensor).unsqueeze(0).float().to(device)
        with torch.no_grad():
            value = self.forward(x).squeeze().item()
        return value

    def evaluate_positions_batch(self, board_tensors: np.ndarray, device: torch.device = None) -> np.ndarray:
        if device is None:
            device = next(self.parameters()).device
        x = torch.from_numpy(board_tensors).float().to(device)
        with torch.no_grad():
            values = self.forward(x).squeeze().cpu().numpy()
        return values


class ValueNetworkWithPolicyHead(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        super(ValueNetworkWithPolicyHead, self).__init__()
        input_size = 12 * 8 * 8
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        value = self.tanh(self.value_head(x))
        policy = self.softmax(self.policy_head(x))
        return value, policy
