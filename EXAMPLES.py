"""
EXAMPLES & DEMONSTRATIONS
Cách sử dụng các modules
"""

# =============================================================================
# EXAMPLE 1: BOARD STATE REPRESENTATION
# =============================================================================

from board_state import BoardState
import chess
import numpy as np

# Tạo board mở đầu
board = chess.Board()

# Chuyển sang tensor
tensor = BoardState.board_to_tensor(board)
print(f"Tensor shape: {tensor.shape}")  # (12, 8, 8)
print(f"Tensor dtype: {tensor.dtype}")  # float32

# Kiểm tra white pawns (plane 0)
print(f"\nWhite Pawns:\n{tensor[0]}")
# [[0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 1. 1. 1. 1. 1. 1. 1.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  ...

# Kiểm tra white pieces tổng cộng
white_pieces = tensor[:6].sum()  # Planes 0-5
print(f"\nTotal white pieces: {white_pieces}")  # 16

# Inverse: tensor → board
board_restored = BoardState.tensor_to_board(tensor)
print(f"\nOriginal FEN: {board.fen()}")
print(f"Restored FEN: {board_restored.fen()}")
print(f"Same: {board.fen() == board_restored.fen()}")  # True


# =============================================================================
# EXAMPLE 2: NEURAL NETWORK
# =============================================================================

from value_network import ValueNetwork
import torch
import numpy as np

# Khởi tạo network
network = ValueNetwork(hidden_size=128, dropout=0.3)
print(f"Network:\n{network}")

# Move to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)

# Tạo dummy board tensors
batch_size = 4
board_tensors = torch.randn(batch_size, 12, 8, 8, device=device)

# Forward pass
output = network(board_tensors)
print(f"\nInput shape: {board_tensors.shape}")   # (4, 12, 8, 8)
print(f"Output shape: {output.shape}")          # (4, 1)
print(f"Output values: {output.squeeze()}")     # [-0.234, 0.567, -0.123, 0.890]

# Đánh giá position từ numpy
board_np = np.random.randn(12, 8, 8).astype(np.float32)
value = network.evaluate_position(board_np, device)
print(f"\nPosition value: {value:.4f}")  # ≈0.5234

# Batch evaluation
batch_np = np.random.randn(8, 12, 8, 8).astype(np.float32)
values = network.evaluate_positions_batch(batch_np, device)
print(f"Batch values: {values}")  # [0.234, -0.456, ...]


# =============================================================================
# EXAMPLE 3: MINIMAX ENGINE
# =============================================================================

from minimax_engine import MinimaxEngine, RandomEngine
from value_network import ValueNetwork
import chess
import torch

# Khởi tạo network
network = ValueNetwork(hidden_size=128)
device = torch.device('cpu')
network.to(device)

# Khởi tạo engines
minimax_engine = MinimaxEngine(network, device=device, max_depth=3)
random_engine = RandomEngine()

# Tạo board
board = chess.Board()
print(f"Starting position")
print(f"FEN: {board.fen()}\n")

# Random move
random_move = RandomEngine.get_best_move(board)
print(f"Random move: {random_move}")

# Minimax move
board_copy = board.copy()
minimax_move, score = minimax_engine.get_best_move_with_score(board_copy)
print(f"Minimax move: {minimax_move}")
print(f"Score: {score:.4f} (positive = white advantage)")
print(f"Nodes evaluated: {minimax_engine.nodes_evaluated}")


# =============================================================================
# EXAMPLE 4: SELF-PLAY
# =============================================================================

from self_play import SelfPlayGame, SelfPlayManager
from value_network import ValueNetwork
from minimax_engine import MinimaxEngine
import torch

# Scenario 1: One game (Random vs Random)
print("=== GAME 1: Random vs Random ===\n")
game = SelfPlayGame(max_moves=50)
result, reason = game.play()

print(f"Result: {result}")  # 1 (white win), 0 (draw), -1 (black win)
print(f"Reason: {reason}")  # 'checkmate', 'stalemate', etc.
print(f"Moves: {len(game.game_data)}")

# Get training data for this game
training_data = game.get_training_data()
print(f"Training samples: {len(training_data)}")
print(f"Sample: state shape = {training_data[0][0].shape}, label = {training_data[0][1]}")

# Scenario 2: Multiple games (Self-play manager)
print("\n=== GAMES 2-21: Multiple self-play ===\n")
manager = SelfPlayManager(device=torch.device('cpu'))
stats = manager.play_games(
    num_games=20,
    white_mode='random',
    black_mode='random',
    max_moves=50
)

print(f"Stats:")
print(f"  Total games: {stats['total_games']}")
print(f"  White wins: {stats['white_wins']}")
print(f"  Black wins: {stats['black_wins']}")
print(f"  Draws: {stats['draws']}")
print(f"  Training samples: {stats['training_samples']}")

# Get training data
board_tensors, labels = manager.get_all_data()
print(f"\nTraining data:")
print(f"  Shape: {board_tensors.shape}")
print(f"  Label distribution: +1={np.sum(labels==1)}, 0={np.sum(labels==0)}, -1={np.sum(labels==-1)}")


# =============================================================================
# EXAMPLE 5: TRAINING
# =============================================================================

from train import ChessTrainer
from value_network import ValueNetwork
import torch
import numpy as np

# Tạo dummy data
board_tensors = np.random.randn(100, 12, 8, 8).astype(np.float32)
labels = np.random.choice([-1.0, 0.0, 1.0], 100).astype(np.float32)

# Split train/val
train_indices = np.random.choice(100, 80, replace=False)
val_indices = np.array([i for i in range(100) if i not in train_indices])

train_data = (board_tensors[train_indices], labels[train_indices])
val_data = (board_tensors[val_indices], labels[val_indices])

# Trainer
network = ValueNetwork(hidden_size=128)
device = torch.device('cpu')
trainer = ChessTrainer(network, device=device, learning_rate=0.001)

# Train
result = trainer.train(
    train_data=train_data,
    val_data=val_data,
    epochs=20,
    batch_size=16,
    early_stopping_patience=5
)

print(f"Training result:")
print(f"  Best epoch: {result['best_epoch']}")
print(f"  Best val loss: {result['best_val_loss']:.6f}")
print(f"  Final train loss: {result['final_train_loss']:.6f}")

# Save model
trainer.save_model('example_model.pth')
print(f"Model saved!")


# =============================================================================
# EXAMPLE 6: GUI GAMEPLAY
# =============================================================================

from gui import create_gui_with_engine
import chess

# Tạo GUI với engine
print("Creating GUI...")
gui = create_gui_with_engine(
    model_path=None,  # Untrained network
    ai_color=chess.BLACK,
    max_depth=2  # Fast for demo
)

print("Starting game...")
# gui.run()  # Uncomment để chơi


# =============================================================================
# EXAMPLE 7: FULL WORKFLOW
# =============================================================================

"""
Workflow đầy đủ:

1. Generate self-play data
   python main.py selfplay --num-games 100 --white-mode random --black-mode random --save-data data.npz

2. Train model
   python main.py train

3. Generate better data using trained model
   python main.py selfplay --num-games 100 --white-mode minimax --black-mode minimax --depth 3 --save-data better_data.npz

4. Re-train model
   python main.py train

5. Play game
   python main.py play --depth 3

6. Analyze position
   python main.py analyze --depth 4 --fen "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
"""


# =============================================================================
# EXAMPLE 8: DEBUGGING
# =============================================================================

# Debug 1: Check board state
from board_state import BoardState
import chess

board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 1")
tensor = BoardState.board_to_tensor(board)

print("Board visualization:")
print(f"White knights (plane 1):\n{tensor[1]}")

# Debug 2: Trace minimax
minimax_engine.get_best_move(board)
print(f"Nodes evaluated: {minimax_engine.nodes_evaluated}")

# Debug 3: Compare evaluations
positions = [
    chess.Board(),  # Starting
    chess.Board("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"),  # After e4 Nc6
]

for board in positions:
    value = network.evaluate_position(BoardState.board_to_tensor(board), device)
    print(f"Position value: {value:.4f}")


# =============================================================================
# EXAMPLE 9: PERFORMANCE TESTING
# =============================================================================

import time
from minimax_engine import MinimaxEngine

# Test different depths
for depth in [1, 2, 3, 4]:
    engine = MinimaxEngine(network, device=device, max_depth=depth)
    board = chess.Board()
    
    start = time.time()
    move = engine.get_best_move(board)
    elapsed = time.time() - start
    
    print(f"Depth {depth}: {elapsed:.3f}s ({engine.nodes_evaluated} nodes)")


# =============================================================================
# EXAMPLE 10: ADVANCED TRAINING
# =============================================================================

"""
Advanced techniques to improve performance:

1. Curriculum learning
   - Start with random vs random
   - Then minimax vs random
   - Finally minimax vs minimax

2. Data augmentation
   - Symmetric positions (rotations)
   - Color swap (white/black)

3. Ensemble
   - Train multiple models
   - Average predictions

4. Fine-tuning
   - Start from pretrained
   - Fine-tune on specific positions

5. Regularization
   - L1/L2 regularization
   - Dropout
   - Early stopping (implemented)
"""

print("\nAll examples completed! 🎉")
