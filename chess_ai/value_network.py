import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class ValueNetwork(nn.Module):
    """
    Value Network để đánh giá vị trí cờ
    
    Input: tensor (12, 8, 8) - 12 planes, 8x8 bàn cờ
    Output: scalar ∈ [-1, 1] - giá trị của vị trí
        ≈ +1: Trắng có lợi
        ≈ 0: Cân bằng
        ≈ -1: Đen có lợi
    """
    
    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        """
        Args:
            hidden_size: Kích thước hidden layer (128 → 64)
            dropout: Dropout rate để tránh overfitting
        """
        super(ValueNetwork, self).__init__()
        
        # Input: 12 * 8 * 8 = 768
        input_size = 12 * 8 * 8
        
        # Flatten + FC layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: tensor shape (batch_size, 12, 8, 8)
            
        Returns:
            tensor shape (batch_size, 1) - giá trị vị trí
        """
        # Flatten: (batch, 12, 8, 8) → (batch, 768)
        x = x.view(x.size(0), -1)
        
        # FC1 + ReLU + Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # FC2 + ReLU + Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # FC3 + Tanh (output ∈ [-1, 1])
        x = self.fc3(x)
        x = self.tanh(x)
        
        return x
    
    def evaluate_position(self, board_tensor: np.ndarray, device: torch.device = None) -> float:
        """
        Đánh giá một vị trí cờ
        
        Args:
            board_tensor: numpy array (12, 8, 8)
            device: torch device (CPU/GPU)
            
        Returns:
            float - giá trị vị trí [-1, 1]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Chuyển sang tensor PyTorch
        x = torch.from_numpy(board_tensor).unsqueeze(0).float().to(device)
        
        # Forward
        with torch.no_grad():
            value = self.forward(x).squeeze().item()
        
        return value
    
    def evaluate_positions_batch(self, board_tensors: np.ndarray, 
                                 device: torch.device = None) -> np.ndarray:
        """
        Đánh giá batch vị trí cờ
        
        Args:
            board_tensors: numpy array (batch_size, 12, 8, 8)
            device: torch device
            
        Returns:
            numpy array (batch_size,) - giá trị các vị trí
        """
        if device is None:
            device = next(self.parameters()).device
        
        x = torch.from_numpy(board_tensors).float().to(device)
        
        with torch.no_grad():
            values = self.forward(x).squeeze().cpu().numpy()
        
        return values


class ValueNetworkWithPolicyHead(nn.Module):
    """
    Extended network với cả value head và policy head (tương lai)
    """
    
    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        super(ValueNetworkWithPolicyHead, self).__init__()
        
        input_size = 12 * 8 * 8
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        
        # Value head
        self.value_head = nn.Linear(64, 1)
        
        # Policy head (output 64 ô)
        self.policy_head = nn.Linear(64, 64)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            value: (batch, 1)
            policy: (batch, 64) - probability distribution
        """
        x = x.view(x.size(0), -1)
        
        # Shared
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Value
        value = self.tanh(self.value_head(x))
        
        # Policy
        policy = self.softmax(self.policy_head(x))
        
        return value, policy


if __name__ == "__main__":
    # Test network
    network = ValueNetwork(hidden_size=128)
    print(f"Network: {network}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 12, 8, 8)  # Batch size 4
    output = network(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze()}")
    
    # Test with numpy
    board_tensor = np.random.randn(12, 8, 8).astype(np.float32)
    value = network.evaluate_position(board_tensor)
    print(f"\nSingle position value: {value:.4f}")
    
    # Test batch
    batch_tensors = np.random.randn(8, 12, 8, 8).astype(np.float32)
    values = network.evaluate_positions_batch(batch_tensors)
    print(f"Batch values shape: {values.shape}")
    print(f"Batch values: {values}")
