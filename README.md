# ♟️ Chess AI - Minimax + Deep Neural Networks

**Chess AI** là một động cơ cờ vua (Chess Engine) hiệu năng cao được phát triển bằng **Python** kết hợp **PyTorch**. Dự án áp dụng các thuật toán cốt lõi trong Lý thuyết Trò chơi (Game Theory) và Học Máy (Machine Learning) để xây dựng một đối thủ máy tính có khả năng thi đấu cạnh tranh. Đây là sản phẩm thuộc đồ án môn học **Nhập môn Trí tuệ Nhân tạo (Introduction to AI)**.

---

## 🎯 Tổng Quan Dự Án

Chess AI kết hợp hai phương pháp tìm kiếm chính:

1. **Minimax + Alpha-Beta Pruning**: Tìm kiếm chiến lược từ trạng thái hiện tại
2. **Deep Neural Networks**: Đánh giá chất lượng từng thế cờ thông qua học máy từ dữ liệu thực

Hệ thống hỗ trợ:
- 🎮 **GUI giao diện**: Chơi với máy tính trực quan trên bàn cờ
- 📊 **Tự huấn luyện (Self-play)**: Cải thiện mô hình thông qua game tự chơi
- 📚 **Huấn luyện từ PGN**: Học từ hàng ngàn ván cờ thực tế
- 📈 **Đánh giá chi tiết**: So sánh với Stockfish, tính toán Elo rating

---

## 🧠 Các Thuật Toán & Tính Năng Nổi Bật

### **Kiến Trúc Neural Network**

#### **FC Network (Fully Connected)**
- **Cấu trúc**: 768 → 128 → 64 → 1 (ReLU, Dropout 0.3, Tanh output)
- **Tham số**: ~107K
- **Ưu điểm**: Nhanh, phù hợp real-time play
- **Độ chính xác ranking**: 79.1% trên validation set

#### **CNN Network (Convolutional)**
- **Cấu trúc**: Conv(12→64) → Conv(64→128) → Conv(128→256) → FC layers
- **Tham số**: ~12.2M (113x lớn hơn FC)
- **Ưu điểm**: Nhận diện patterns không gian, độ chính xác cao
- **Độ chính xác ranking**: 81.1% trên validation set
- **Loss cải thiện**: 10x tốt hơn FC (0.001-0.002 vs 0.015-0.020)

### **Thuật Toán Tìm Kiếm (Search Algorithm)**

- **Minimax + Alpha-Beta Pruning**: Cắt tỉa các nhánh không cần thiết, giảm nodes evaluate 50-70%
- **Iterative Deepening**: Quản lý thời gian suy nghĩ hiệu quả
- **Move Ordering Heuristics**: Tối ưu hóa thứ tự duyệt nước đi
  - Material Value (Material Balance)
  - Hybrid Evaluation: 70% Neural Network + 30% Material Value

### **Dữ Liệu Training**

| Dataset | Size | Samples | Mô Tả |
|---------|------|---------|-------|
| pgn_training_data.npz | 531 MB | 180,822 | Full dataset từ Chess.com |
| pgn_training_data_perfectly_balanced.npz | 2.3 MB | 55,013 | Balanced (cân bằng win/loss/draw) |
| pgn_training_data_balanced.npz | 1.9 MB | 45,898 | Balanced variant |

### **Huấn Luyện**

**Default Configuration** (từ `train_from_pgn()`):
- **Epochs**: 200
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam (L2 regularization = 1e-5)
- **Loss Function**: MSE / MarginRankingLoss (ranking training)
- **Early Stopping Patience**: 30 epochs
- **Dropout**: 0.3

---

## ⚙️ Yêu Cầu Hệ Thống

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **python-chess**: 1.99+
- **NumPy**: 1.24+
- **Matplotlib**: 3.7+ (cho visualization)
- **Stockfish**: (Tùy chọn) Dành cho đánh giá vs Stockfish

**Dependencies** (xem `requirements.txt`):
```
torch==2.0.0
numpy==1.24.0
python-chess==1.99
pygame==2.2.0
matplotlib==3.7.0
stockfish==3.28.0
requests==2.31.0
trueskill==0.4.5
```

---

## 🛠️ Hướng Dẫn Cài Đặt

### **1. Clone Dự Án**

```bash
git clone <https://github.com/jokercode258/Project1>
```

### **2. Cài Đặt Dependencies**

```bash
pip install -r requirements.txt
```

### **3. (Tùy Chọn) Cài Đặt Stockfish**

Để sử dụng chức năng đánh giá vs Stockfish:

**Windows**:
```bash
# Tải từ https://stockfishchess.org/download/
# Cài đặt hoặc giải nén vào đâu đó, sau đó:
set STOCKFISH_PATH=C:\path\to\stockfish.exe
```

**Linux/macOS**:
```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# macOS
brew install stockfish

# Xác định đường dẫn:
which stockfish
```
---

## 📁 Cấu Trúc Dự Án

```
chess_ai/
├── board/                      # Board state & representation
│   ├── board_state.py
│   └── __init__.py
├── data_processing/            # PGN processing & data handling
│   ├── pgn_downloader.py       # Chess.com downloader
│   ├── pgn_processor.py        # PGN parser
│   └── __init__.py
├── engine/                     # Game AI & evaluation
│   ├── minimax_engine.py       # Minimax + Alpha-Beta Pruning
│   ├── tactical_evaluator.py
│   ├── tactical_value_function.py
│   └── __init__.py
├── network/                    # Neural Networks
│   ├── value_network.py        # FC & CNN architectures
│   └── __init__.py
└── __init__.py

training/
├── pgn/                        # Training from PGN files
│   ├── download.py
│   ├── pgn_parser.py
│   ├── train.py                # Main training pipeline
│   └── __init__.py
├── self_play/                  # Self-play training
│   ├── self_play.py
│   ├── train.py
│   └── __init__.py
└── __init__.py

scripts/                        # Utility scripts
├── evaluate_vs_random.py       # Evaluation vs random player
├── evaluate_vs_stockfish.py    # Evaluation vs Stockfish
├── compare_models.py           # Model comparison
├── plot_evaluation_results.py  # Visualization
└── __init__.py

gui_module/                     # GUI for playing
├── gui.py
└── __init__.py

data/
├── models/                     # Trained models
│   ├── pgn/                    # Models from PGN training
│   │   ├── best_model.pth
│   │   ├── improved_network.pth (CNN)
│   │   └── chess_value_network.pth (FC)
│   └── self_play/              # Models from self-play
├── datasets/                   # Training datasets
│   └── pgn/                    # PGN datasets (.npz files)
└── evaluation/                 # Evaluation results
    ├── random/                 # vs Random results
    └── stockfish/              # vs Stockfish results

pgn_files/                     # Downloaded PGN files from Chess.com

main.py                        # Main CLI interface
requirements.txt               # Dependencies
README.md                      # This file
```
---

## 🎮 Hướng Dẫn Sử Dụng

Dự án cung cấp một **Command-Line Interface (CLI)** với nhiều chế độ khác nhau:

```bash
python main.py <command> [options]
```

### **1. Chơi với AI (GUI Mode)** 🎮

```bash
python main.py play --depth 3 --player-color white
```

**Tùy chọn**:
- `--depth`: Độ sâu Minimax (mặc định: 3)
- `--player-color`: Màu của người chơi - `white` hoặc `black` (mặc định: white)
- `--model`: Đường dẫn tới model weights (tùy chọn)

**Ví dụ**:
```bash
# Chơi với màu đen, depth=4
python main.py play --depth 4 --player-color black

# Chơi với model cụ thể
python main.py play --model ./data/models/pgn/improved_network.pth --depth 3
```

### **2. Huấn Luyện Từ PGN Files** 📚

```bash
python main.py train-pgn --pgn-source ./pgn_files --improved-network
```

**Tùy chọn**:
- `--pgn-source`: Đường dẫn tới PGN file hoặc directory chứa PGN files (bắt buộc)
- `--improved-network`: Sử dụng CNN Network thay vì FC Network
- `--epochs`: Số epochs (mặc định: 200)
- `--batch-size`: Batch size (mặc định: 64)
- `--max-positions`: Số positions tối đa để extract (mặc định: 100000)
- `--patience`: Early stopping patience (mặc định: 30)
- `--stockfish-path`: Đường dẫn tới Stockfish (nếu không có, sẽ tìm tự động)

**Ví dụ**:
```bash
# Huấn luyện CNN từ PGN files
python main.py train-pgn --pgn-source ./pgn_files --improved-network --epochs 200 --batch-size 64

# Huấn luyện FC network
python main.py train-pgn --pgn-source ./pgn_files --epochs 150 --batch-size 128
```

### **3. Tải PGN Files Từ Chess.com** 📥

```bash
python main.py download --output-dir ./pgn_files
```

**Tùy chọn**:
- `--output-dir`: Directory để lưu PGN files (mặc định: ./pgn_files)
- `--player`: Tải games từ một player cụ thể (ví dụ: "nakamura", "carlsen")

**Ví dụ**:
```bash
# Tải games từ Hikaru Nakamura
python main.py download --output-dir ./pgn_files --player nakamura

# Tải games từ top GMs
python main.py download --output-dir ./pgn_files
```

### **4. Self-Play Mode** 🤖🤖

```bash
python main.py selfplay --num-games 50 --depth 3
```

**Tùy chọn**:
- `--num-games`: Số games để chơi (mặc định: 20)
- `--white-mode`: Engine mode cho White - `random` hoặc `minimax` (mặc định: random)
- `--black-mode`: Engine mode cho Black - `random` hoặc `minimax` (mặc định: random)
- `--max-moves`: Số nước tối đa per game (mặc định: 100)
- `--depth`: Minimax depth (mặc định: 3)
- `--model`: Đường dẫn tới model weights
- `--save-data`: Lưu training data tới file (tùy chọn)

**Ví dụ**:
```bash
# Self-play: Minimax (depth=4) vs Random, 100 games
python main.py selfplay --num-games 100 --white-mode minimax --black-mode random --depth 4 --save-data ./training_data.npz

# Self-play: AI vs AI
python main.py selfplay --num-games 50 --white-mode minimax --black-mode minimax --depth 3 --model ./data/models/pgn/improved_network.pth
```

### **5. Đánh Giá vs Random Player** 📊

```bash
python scripts/evaluate_vs_random.py
```

Kết quả sẽ được lưu vào `data/evaluation/random/evaluation_log.csv`.

### **6. Đánh Giá vs Stockfish** 🏆

```bash
python scripts/evaluate_vs_stockfish.py --stockfish /path/to/stockfish --model ./data/models/pgn/improved_network.pth --games 100 --depth 3
```

**Tùy chọn**:
- `--stockfish`: Đường dẫn tới Stockfish (bắt buộc)
- `--model`: Đường dẫn tới model
- `--games`: Số games để chơi (mặc định: 100)
- `--depth`: Minimax depth (mặc định: 3)
- `--time`: Seconds per move (mặc định: 0.5)
- `--opponent-elo`: Assumed Elo của Stockfish (mặc định: 3500)
- `--skill`: Stockfish Skill Level 0-20 (mặc định: 20)
- `--pgn-out`: Output PGN file (tùy chọn)

**Ví dụ**:
```bash
python scripts/evaluate_vs_stockfish.py --stockfish /usr/bin/stockfish --model ./data/models/pgn/improved_network.pth --games 100 --depth 3 --skill 10 --pgn-out results.pgn
```

### **7. Phân Tích Vị Trí Cụ Thể** 🔍

```bash
python main.py analyze --model ./data/models/pgn/improved_network.pth --depth 4
```

**Tùy chọn**:
- `--model`: Đường dẫn tới model
- `--depth`: Minimax depth (mặc định: 3)
- `--fen`: FEN string (mặc định: vị trí khởi đầu)

**Ví dụ**:
```bash
# Phân tích vị trí sau e4 e5 Nf3
python main.py analyze --depth 4 --fen "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 2"
```

---

## 🔄 Workflow Típico

### **Huấn Luyện Từ Scratch**

```bash
# Bước 1: Tải PGN files từ Chess.com
python main.py download --output-dir ./pgn_files

# Bước 2: Huấn luyện CNN model từ PGN files
python main.py train-pgn --pgn-source ./pgn_files --improved-network --epochs 200 --batch-size 64

# Bước 3: Đánh giá model
python scripts/evaluate_vs_random.py
python scripts/evaluate_vs_stockfish.py --stockfish /path/to/stockfish --games 100

# Bước 4: Chơi game
python main.py play --depth 3

```
