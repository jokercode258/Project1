import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import os
import logging
import shutil
from pathlib import Path

from chess_ai.network.value_network import ValueNetwork, ImprovedValueNetwork
from .pgn_parser import PGNProcessor

logger = logging.getLogger(__name__)


def find_stockfish_path() -> Optional[str]:
    # 1. Kiểm tra environment variable
    env_path = os.getenv('STOCKFISH_PATH')
    if env_path and os.path.exists(env_path):
        logger.info(f"Found Stockfish from STOCKFISH_PATH: {env_path}")
        return env_path
    
    # 2. Kiểm tra Windows common paths
    windows_paths = [
        r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\Program Files\Stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\Program Files (x86)\Stockfish\stockfish-windows-x86-64-avx2.exe",
        "stockfish-windows-x86-64-avx2.exe",
        "stockfish.exe",
    ]
    
    for path in windows_paths:
        if os.path.exists(path):
            logger.info(f"Found Stockfish at: {path}")
            return path
    
    # 3. Tìm trong PATH
    stockfish_exe = shutil.which('stockfish') or shutil.which('stockfish.exe')
    if stockfish_exe:
        logger.info(f"Found Stockfish in PATH: {stockfish_exe}")
        return stockfish_exe
    
    logger.warning("Stockfish executable not found. Set STOCKFISH_PATH environment variable or ensure it's in PATH")
    return None


class ChessTrainer:
    """
    Trainer cho Value Network
    """
    
    def __init__(self, network: ValueNetwork, device: torch.device = None,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 output_dir: str = './data/models/pgn'):
        """
        Args:
            network: ValueNetwork model
            device: torch device (CPU/GPU)
            learning_rate: Learning rate cho optimizer
            weight_decay: L2 regularization
            output_dir: Directory để lưu models
        """
        self.network = network
        self.device = device if device else torch.device('cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
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
            board_tensors = board_tensors.to(self.device).float()
            labels = labels.to(self.device).float().unsqueeze(1)  # Shape: (batch, 1)
            
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
                board_tensors = board_tensors.to(self.device).float()
                labels = labels.to(self.device).float().unsqueeze(1)
                
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
            torch.from_numpy(board_tensors).float(),
            torch.from_numpy(labels).float()
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None:
            val_board_tensors, val_labels = val_data
            val_dataset = TensorDataset(
                torch.from_numpy(val_board_tensors).float(),
                torch.from_numpy(val_labels).float()
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
                    checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
                    self._save_checkpoint(checkpoint_path)
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_type': self.network.__class__.__name__,
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'network_type': self.network.__class__.__name__,
        }, filepath)
        print(f"Lưu model vào {filepath} (Network: {self.network.__class__.__name__})")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Legacy: plain state dict
            self.network.load_state_dict(checkpoint)
        print(f"Load model từ {filepath} (Network: {self.network.__class__.__name__})")


def train_from_pgn(pgn_source: str, output_dir: str = './data/models/pgn', 
                   use_improved_network: bool = True,
                   max_positions: int = 100000,
                   epochs: int = 200,
                   batch_size: int = 64,
                   early_stopping_patience: int = 30,
                   stockfish_path: Optional[str] = None) -> Tuple[ValueNetwork, ChessTrainer]:
    """
    Train model từ PGN files sử dụng Stockfish evaluation
    
    Args:
        pgn_source: Đường dẫn đến PGN file hoặc directory chứa PGN files
        output_dir: Directory để lưu model (mặc định: ./data/models/pgn)
        use_improved_network: Dùng ImprovedValueNetwork (CNN) hoặc ValueNetwork
        max_positions: Số positions tối đa để extract
        epochs: Số epochs
        batch_size: Batch size
        early_stopping_patience: Patience cho early stopping
        stockfish_path: Đường dẫn tới Stockfish executable (nếu None, sẽ tìm tự động)
        
    Returns:
        (network, trainer)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng device: {device}\n")
    
    # Tìm Stockfish nếu chưa có path
    if stockfish_path is None:
        stockfish_path = find_stockfish_path()
    
    if stockfish_path:
        print(f"Sử dụng Stockfish: {stockfish_path}\n")
    else:
        print("⚠️  Stockfish không tìm được. Training sẽ sử dụng heuristic evaluation thay vì Stockfish.\n")
    
    # Step 1: Parse PGN files
    print("=" * 50)
    print("BƯỚC 1: PARSE PGN FILES")
    print("=" * 50 + "\n")
    
    processor = PGNProcessor(stockfish_path=stockfish_path)
    
    if os.path.isdir(pgn_source):
        print(f"Parsing directory: {pgn_source}")
        board_tensors, labels = processor.parse_pgn_directory(
            pgn_source, 
            max_positions=max_positions
        )
    else:
        print(f"Parsing file: {pgn_source}")
        board_tensors, labels = processor.parse_pgn_file(pgn_source)
    
    if len(board_tensors) == 0:
        print("❌ Không thể parse PGN files!")
        return None, None
    
    print(f"\n✅ Extracted {len(board_tensors)} positions")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Min label: {labels.min():.4f}, Max: {labels.max():.4f}")
    
    # Lưu training data
    os.makedirs('./data/datasets/pgn', exist_ok=True)
    train_data_path = './data/datasets/pgn/pgn_training_data.npz'
    processor.save_training_data(board_tensors, labels, train_data_path)
    
    # Step 2: Tạo network
    print("\n" + "=" * 50)
    print("BƯỚC 2: KHỞI TẠO NETWORK")
    print("=" * 50 + "\n")
    
    if use_improved_network:
        network = ImprovedValueNetwork(dropout=0.3)
        print("Sử dụng ImprovedValueNetwork (CNN)")
    else:
        network = ValueNetwork(hidden_size=128, dropout=0.3)
        print("Sử dụng ValueNetwork (FC)")
    
    print(f"\nNetwork architecture:\n{network}")
    
    # Tính số parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Step 3: Training
    print("\n" + "=" * 50)
    print("BƯỚC 3: TRAINING")
    print("=" * 50 + "\n")
    
    # Split data: 80% train, 20% val
    n_samples = len(labels)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = (board_tensors[train_indices], labels[train_indices])
    val_data = (board_tensors[val_indices], labels[val_indices])
    
    print(f"Train set: {len(train_indices)} mẫu")
    print(f"Val set: {len(val_indices)} mẫu")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}\n")
    
    # Trainer
    trainer = ChessTrainer(network, device=device, learning_rate=0.001, weight_decay=1e-5, output_dir=output_dir)
    
    # Train
    result = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience
    )
    
    print(f"\n" + "=" * 50)
    print("TRAINING HOÀN THÀNH")
    print("=" * 50)
    print(f"\nKết quả:")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val loss: {result['best_val_loss']:.6f}")
    print(f"  Final train loss: {result['final_train_loss']:.6f}")
    
    # Lưu model
    model_name = 'improved_network.pth' if use_improved_network else 'chess_value_network.pth'
    model_path = os.path.join(output_dir, model_name)
    trainer.save_model(model_path)
    
    # Vẽ biểu đồ
    plot_name = 'improved_training_loss.png' if use_improved_network else 'training_loss.png'
    plot_path = os.path.join(output_dir, plot_name)
    trainer.plot_loss(plot_path)
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Loss plot saved: {plot_path}")
    print(f"\n💡 Để load model này, dùng:")
    if use_improved_network:
        print(f"   python main.py play --model {model_path} --depth 3 --improved-network")
    else:
        print(f"   python main.py play --model {model_path} --depth 3")
    
    return network, trainer


if __name__ == "__main__":
    train_from_pgn(
        pgn_source='./pgn_files',
        use_improved_network=True,
        max_positions=100000,
        epochs=200
    )
