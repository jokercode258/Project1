"""
QUICK START GUIDE
"""

# =============================================================================
# 1. CHUẨN BỊ MÔI TRƯỜNG
# =============================================================================

# Cài đặt dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import chess; import pygame; print('✅ All dependencies installed')"


# =============================================================================
# 2. TRAIN MODEL LẦN ĐẦU
# =============================================================================

# Tùy chọn A: Full pipeline (self-play + training)
python main.py train --output-dir ./models

# Tùy chọn B: Chỉ tạo self-play data
python main.py selfplay --num-games 20 --white-mode random --black-mode random --save-data data.npz


# =============================================================================
# 3. CHƠI GAME VỚI AI
# =============================================================================

# A. Chơi với Trắng (bạn đi trước)
python main.py play --player-color white --depth 3

# B. Chơi với Đen (AI đi trước)
python main.py play --player-color black --depth 4

# Điều khiển:
#   - Click chuột trái: Chọn quân
#   - Kéo đến ô khác: Di chuyển
#   - Click phải: Undo
#   - R: Reset game
#   - Q: Quit


# =============================================================================
# 4. PHÂN TÍCH VỊ TRỊ
# =============================================================================

# Analyze vị trí mở đầu
python main.py analyze --depth 3

# Analyze vị trí custom (FEN format)
python main.py analyze --depth 4 --fen "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"


# =============================================================================
# 5. GENERATE MỌI DATA
# =============================================================================

# Minimax vs Random (để train NN học từ minimax)
python main.py selfplay --num-games 50 --white-mode minimax --black-mode random --depth 3 --save-data minimax_vs_random.npz

# Minimax vs Minimax (data tốt nhất)
python main.py selfplay --num-games 100 --white-mode minimax --black-mode minimax --depth 3 --save-data minimax_vs_minimax.npz


# =============================================================================
# 6. ADVANCED USAGE
# =============================================================================

# Thay đổi độ sâu Minimax (lớn = mạnh hơn nhưng chậm hơn)
python main.py play --depth 1    # Nhanh, yếu
python main.py play --depth 3    # Cân bằng
python main.py play --depth 5    # Mạnh, chậm

# Sử dụng GPU (nếu có)
# (Code tự động detect GPU, không cần config)


# =============================================================================
# 7. WORKFLOW TRAINING LOOP
# =============================================================================

# Step 1: Tạo training data
python main.py selfplay --num-games 50 --white-mode random --black-mode random --save-data initial_data.npz

# Step 2: Train model đầu tiên
python main.py train

# Step 3: Play một vài game để cảm nhận
python main.py play --depth 2

# Step 4: Generate more quality data bằng trained model
python main.py selfplay --num-games 100 --white-mode minimax --black-mode minimax --depth 3 --save-data improved_data.npz

# Step 5: Re-train model với dữ liệu tốt hơn
python main.py train

# Repeat 3-5 để tiếp tục improve


# =============================================================================
# 8. DEBUG & TESTING
# =============================================================================

# Test board state representation
python -c "
from board_state import BoardState
import chess
board = chess.Board()
tensor = BoardState.board_to_tensor(board)
print(f'Tensor shape: {tensor.shape}')
print(f'White pawns plane: {tensor[0]}')
"

# Test neural network
python -c "
from value_network import ValueNetwork
import numpy as np
network = ValueNetwork()
dummy_board = np.random.randn(12, 8, 8).astype(np.float32)
value = network.evaluate_position(dummy_board)
print(f'Position value: {value:.4f}')
"

# Test minimax engine
python -c "
from minimax_engine import MinimaxEngine
from value_network import ValueNetwork
import chess
network = ValueNetwork()
engine = MinimaxEngine(network, max_depth=2)
board = chess.Board()
move = engine.get_best_move(board)
print(f'Best move: {move}')
"

# Test self-play
python -c "
from self_play import SelfPlayManager
manager = SelfPlayManager()
stats = manager.play_games(num_games=5, white_mode='random', black_mode='random')
print(stats)
"


# =============================================================================
# 9. FILE STRUCTURE
# =============================================================================

project1/
├── board_state.py          # Biểu diễn state (12x8x8)
├── value_network.py        # Neural Network
├── minimax_engine.py       # Minimax + Alpha-Beta
├── self_play.py            # Self-play generator
├── train.py                # Training loop
├── gui.py                  # Pygame GUI
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── QUICKSTART.md           # File này
└── models/                 # Output directory
    ├── chess_value_network.pth
    ├── best_model.pth
    ├── training_loss.png
    └── training_data.npz


# =============================================================================
# 10. TROUBLESHOOTING
# =============================================================================

# Lỗi: "ModuleNotFoundError: No module named 'chess'"
# Giải pháp: pip install python-chess

# Lỗi: "No module named 'pygame'"
# Giải pháp: pip install pygame

# Lỗi: "CUDA out of memory"
# Giải pháp: Giảm batch size trong train.py hoặc dùng CPU

# Lỗi: "AI move quá chậm"
# Giải pháp: Giảm depth (--depth 2 thay vì 4)

# Lỗi: "pygame window không mở"
# Giải pháp: Đảm bảo X11 display hoặc sử dụng headless mode


# =============================================================================
# 11. TIPS & TRICKS
# =============================================================================

# Xem game history
python -c "
import chess
board = chess.Board()
print('Move history:', board.move_stack)
"

# Export game to PGN
with open('game.pgn', 'w') as f:
    f.write(str(game))

# Calculate eval from FEN
python main.py analyze --fen 'r1bqkb1r/pppppppp/2n2n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1'

# Batch training data
python main.py selfplay --num-games 1000 --white-mode minimax --black-mode minimax --depth 3 --save-data large_dataset.npz


# =============================================================================
# 12. PERFORMANCE NOTES
# =============================================================================

Depth   Speed       Quality     Recommended for
1       Very Fast   Weak        Testing
2       Fast        Decent      Casual play
3       Medium      Good        Balanced (default)
4       Slow        Strong      Serious play
5+      Very Slow   Very Strong  Analysis only

GPU acceleration: ~3-5x faster than CPU
Alpha-Beta: ~10-100x speedup vs pure minimax


# =============================================================================
# DONE! 🎉
# =============================================================================

"""

# Ứng dụng AI Chess của bạn đã sẵn sàng!
# Bước tiếp theo: python main.py train
