import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from chess_ai.value_network import ValueNetwork
from chess_ai.self_play import SelfPlayManager
import os


class ChessTrainer:
    """
    Trainer cho Value Network
    """
    
    def __init__(self, network: ValueNetwork, device: torch.device = None,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Args:
            network: ValueNetwork model
            device: torch device (CPU/GPU)
            learning_rate: Learning rate cho optimizer
            weight_decay: L2 regularization
        """
        self.network = network
        self.device = device if device else torch.device('cpu')
        self.network.to(self.device)
        
        # Loss function và Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(network.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train một epoch
        
        Returns:
            Trung bình loss trên epoch
        """
        self.network.train()
        total_loss = 0.0
        
        for board_tensors, labels in train_loader:
            board_tensors = board_tensors.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Shape: (batch, 1)
            
            # Forward
            predictions = self.network(board_tensors)
            loss = self.criterion(predictions, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model
        
        Returns:
            Validation loss
        """
        self.network.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for board_tensors, labels in val_loader:
                board_tensors = board_tensors.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                predictions = self.network(board_tensors)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32,
              early_stopping_patience: int = 5) -> dict:
        """
        Huấn luyện model
        
        Args:
            train_data: (board_tensors, labels)
            val_data: (board_tensors, labels) cho validation
            epochs: Số epoch
            batch_size: Batch size
            early_stopping_patience: Số epoch không cải thiện để dừng
            
        Returns:
            {'best_epoch': int, 'best_val_loss': float, ...}
        """
        # Tạo DataLoader
        board_tensors, labels = train_data
        train_dataset = TensorDataset(
            torch.from_numpy(board_tensors),
            torch.from_numpy(labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None:
            val_board_tensors, val_labels = val_data
            val_dataset = TensorDataset(
                torch.from_numpy(val_board_tensors),
                torch.from_numpy(val_labels)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        patience_counter = 0
        best_epoch = 0
        
        print(f"Bắt đầu training cho {epochs} epochs")
        print(f"Train set: {len(train_dataset)} mẫu")
        if val_loader:
            print(f"Val set: {len(val_dataset)} mẫu\n")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    # Lưu best model
                    self._save_checkpoint(f'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping tại epoch {epoch}")
                        break
            
            # In thông tin
            if (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}"
                if val_loss:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None
        }
    
    def _save_checkpoint(self, filepath: str):
        """Lưu checkpoint"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def plot_loss(self, save_path: str = None):
        """Vẽ biểu đồ loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.title('Training Progress')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Lưu biểu đồ vào {save_path}")
        else:
            plt.show()
    
    def save_model(self, filepath: str):
        """Lưu model weights"""
        torch.save(self.network.state_dict(), filepath)
        print(f"Lưu model vào {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Load model từ {filepath}")


def full_training_pipeline(output_dir: str = './data'):
    """
    Full pipeline: Self-play → Training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}\n")
    
    # Self-play để tạo training data
    print("=" * 50)
    print("BƯỚC 1: SELF-PLAY")
    print("=" * 50)
    
    manager = SelfPlayManager(device=device)
    stats = manager.play_games(num_games=20, white_mode='random', black_mode='random', max_moves=50)
    
    print(f"\nSelf-play stats:")
    print(f"  Trắng thắng: {stats['white_wins']}")
    print(f"  Đen thắng: {stats['black_wins']}")
    print(f"  Hòa: {stats['draws']}")
    print(f"  Training samples: {stats['training_samples']}")
    
    # Lưu training data
    train_data_path = os.path.join(output_dir, 'training_data.npz')
    manager.save_training_data(train_data_path)
    
    # Tạo network
    print("\n" + "=" * 50)
    print("BƯỚC 2: KHỞI TẠO NETWORK")
    print("=" * 50)
    
    network = ValueNetwork(hidden_size=128, dropout=0.3)
    print(f"Network:\n{network}")
    
    # Training
    print("\n" + "=" * 50)
    print("BƯỚC 3: TRAINING")
    print("=" * 50 + "\n")
    
    # Split data: 80% train, 20% val
    board_tensors, labels = manager.get_all_data()
    n_samples = len(labels)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = (board_tensors[train_indices], labels[train_indices])
    val_data = (board_tensors[val_indices], labels[val_indices])
    
    # Trainer
    trainer = ChessTrainer(network, device=device, learning_rate=0.001)
    
    # Train
    result = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=100,
        batch_size=32,
        early_stopping_patience=10
    )
    
    print(f"\nTraining hoàn thành:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val loss: {result['best_val_loss']:.6f}")
    
    # Lưu model
    model_path = os.path.join(output_dir, 'chess_value_network.pth')
    trainer.save_model(model_path)
    
    # Vẽ biểu đồ
    plot_path = os.path.join(output_dir, 'training_loss.png')
    trainer.plot_loss(plot_path)
    
    print(f"\nHoàn thành training pipeline!")
    print(f"Model saved: {model_path}")
    
    return network, trainer


if __name__ == "__main__":
    # Chạy full pipeline
    network, trainer = full_training_pipeline()
