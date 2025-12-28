# AI Chess

Hệ thống AI chơi cờ tích hợp đầy đủ với Minimax, Neural Network, Self-play, và GUI Pygame.


# Cấu trúc dự án

project1/
|
├── assets/               (PNG hình quân cờ)
├── data/                 (THƯ MỤC DỮ LIỆU MỚI)
│   ├── training_data.npz (dữ liệu self-play)
│   ├── chess_value_network.pth (model weights)
│   └── best_model.pth (checkpoint)
├── chess_ai/             (AI core modules)
│   ├── __init__.py
│   ├── board_state.py    (Biểu diễn trạng thái (12 planes))
│   ├── value_network.py  (Neural Network) 
│   ├── minimax_engine.py (Minimax + Alpha-Beta + NN)
│   └── self_play.py      (Self-play để tạo training data)
├── gui_module/           (GUI interface)
│   ├── __init__.py
│   └── gui.py            (Giao diện Pygame)
├── training/             (Training pipeline)
│   ├── __init__.py
│   └── train.py          (Training loop)
├── main.py               (Entry point)
└── requirements.txt

## 6 Bước chính

# BƯỚC 1: State Representation (board_state.py)
- Chuyển bàn cờ → tensor 12×8×8
- 12 planes: 6 quân trắng + 6 quân đen
- Mapping tọa độ: a8→(0,0), h1→(7,7)

# BƯỚC 2: Self-Play (self_play.py)
- AI tự chơi với chính nó
- Ghi lại mọi state + kết quả
- Tạo training dataset

# BƯỚC 3: Neural Network (value_network.py)
- Input: 768 (12×8×8)
- Hidden: 128 → 64
- Output: 1 (value ∈ [-1, 1])
- Activation: ReLU (hidden), Tanh (output)

# BƯỚC 4: Training (train.py)
- Loss function: MSELoss
- Optimizer: Adam
- Training loop with validation
- Early stopping + checkpointing

# BƯỚC 5: Minimax + NN (minimax_engine.py)
- Minimax duyệt cây
- Alpha-Beta pruning để tối ưu
- NN đánh giá node lá
- Không sinh nước đi (Minimax chủ trương)

# BƯỚC 6: GUI (gui.py)
- Vẽ bàn cờ 8×8 với Pygame
- Click để chọn quân
- Highlight nước đi hợp lệ
- AI tự động đi

## Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
python main.py train --output-dir ./models
```

Sẽ:
1. Chơi 20 self-play games
2. Tạo training data
3. Train network 100 epochs
4. Lưu model tốt nhất

### 3. Chơi game

```bash
python main.py play --model ./models/chess_value_network.pth --depth 3 --player-color white
```

Điều khiển:
- **Click trái**: Chọn quân
- **Click phải**: Undo nước đi
- **R**: Reset game
- **Q**: Quit

### 4. Self-play (generate more data)

```bash
python main.py selfplay --num-games 50 --white-mode minimax --black-mode random --save-data training_data.npz
```

### 5. Analyze position

```bash
python main.py analyze --model ./models/chess_value_network.pth --depth 3 --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

## Kiến trúc tổng quát

```
Người chơi
   ↓
Pygame GUI (gui.py)
   ↓
Chess Board (chess.Board)
   ↓
Minimax Engine (minimax_engine.py)
   ├─ Duyệt cây (Minimax)
   ├─ Alpha-Beta Pruning
   └─ Evaluate node lá bằng NN
       └─ Value Network (value_network.py)
           └─ Input: board_state.py (12×8×8)
```

## Ví dụ workflow

**Bước 1: Train lần đầu**
```bash
python main.py train
```

**Bước 2: Chơi game**
```bash
python main.py play --depth 3
```

**Bước 3: Tạo more data để train lại**
```bash
python main.py selfplay --num-games 100 --white-mode minimax --black-mode minimax --save-data more_data.npz
```

**Bước 4: Train thêm (fine-tune)**
```bash
python main.py train --output-dir ./models
```

## Key concepts

### Board Representation
- 12 planes: P,N,B,R,Q,K (white) + p,n,b,r,q,k (black)
- Mỗi plane 8×8 binary matrix
- Efficient cho NN processing

### Minimax + NN Integration
- Minimax tìm kiếm (với Alpha-Beta)
- NN không sinh nước đi, chỉ đánh giá
- Kết hợp = AI có "suy tư" + "kỹ năng"

### Self-play Learning
- Game 1: Random vs Random → học cơ bản
- Game N: Minimax vs Minimax → học chiến lược nâng cao
- Cải tiến dần dần thông qua các lần lặp lại

## Training progression

```
Epoch 1   : loss=0.8523  val_loss=0.8412  (random knowledge)
Epoch 10  : loss=0.4231  val_loss=0.4189  (learning basic tactics)
Epoch 50  : loss=0.1234  val_loss=0.1289  (understanding positions)
Epoch 100 : loss=0.0234  val_loss=0.0245  (refined evaluation)
```



### Inspect position
```python
from board_state import BoardState
board_tensor = BoardState.board_to_tensor(board)
print(board_tensor.shape)  # (12, 8, 8)
```

### Test network
```python
from value_network import ValueNetwork
network = ValueNetwork()
value = network.evaluate_position(board_tensor)  # -1 to 1
```

### Check AI move
```python
engine = MinimaxEngine(network, max_depth=4)
move, score = engine.get_best_move_with_score(board)
print(f"Move: {move}, Score: {score:.4f}")
```

##  References

- **Minimax**: Classic game theory algorithm
- **Alpha-Beta Pruning**: Optimization for game trees
- **Self-play**: AlphaGo methodology
- **Value Networks**: Neural network as evaluator
- **Python-Chess**: Chess logic library
- **PyTorch**: Deep learning framework
- **Pygame**: Game graphics


