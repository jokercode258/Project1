"""
AI CHESS SYSTEM - COMPLETE IMPLEMENTATION
Minimax + Neural Network + Self-play + GUI
"""

# =============================================================================
# 📁 PROJECT STRUCTURE
# =============================================================================

AI Chess System/
│
├── 📄 README.md                 ← START HERE! Project overview & usage
├── 📄 QUICKSTART.md             ← Quick commands to get started
├── 📄 ARCHITECTURE.md           ← Detailed technical architecture
├── 📄 EXAMPLES.py               ← Code examples for each component
│
├── 🔷 BƯỚC 1: STATE REPRESENTATION
│   └── 📄 board_state.py        (12x8x8 tensor representation)
│
├── 🔷 BƯỚC 2: SELF-PLAY DATA GENERATION
│   └── 📄 self_play.py          (AI plays against itself)
│
├── 🔷 BƯỚC 3: NEURAL NETWORK DESIGN
│   └── 📄 value_network.py      (ValueNetwork, 768→128→64→1)
│
├── 🔷 BƯỚC 4: TRAINING LOOP
│   └── 📄 train.py              (PyTorch training with MSE + Adam)
│
├── 🔷 BƯỚC 5: MINIMAX + NN INTEGRATION
│   └── 📄 minimax_engine.py     (Minimax + Alpha-Beta + NN evaluation)
│
├── 🔷 BƯỚC 6: INTERACTIVE GUI
│   └── 📄 gui.py                (Pygame GUI for playing)
│
├── 🎮 MAIN ENTRY POINT
│   └── 📄 main.py               (CLI with subcommands)
│
└── 📦 DEPENDENCIES
    └── 📄 requirements.txt       (Python packages)


# =============================================================================
# 🎯 QUICK START
# =============================================================================

1. Install dependencies:
   $ pip install -r requirements.txt

2. Train model:
   $ python main.py train

3. Play game:
   $ python main.py play

4. Analyze position:
   $ python main.py analyze


# =============================================================================
# 📊 FILE DESCRIPTIONS
# =============================================================================

🔷 BƯỚC 1: board_state.py
─────────────────────────────────────
Purpose: Convert chess board ↔ tensor (12×8×8)

Key functions:
  • board_to_tensor(board) → numpy array
    - Input: chess.Board object
    - Output: (12, 8, 8) float32 tensor
    - Planes 0-5: White pieces (P,N,B,R,Q,K)
    - Planes 6-11: Black pieces (p,n,b,r,q,k)

  • tensor_to_board(tensor) → chess.Board
    - Inverse transformation

  • get_game_result(board) → float
    - Returns 1.0, 0.0, -1.0 for W/D/L

  • get_legal_moves_tensor(board) → numpy array
    - Mask of legal move destinations

Example:
  tensor = BoardState.board_to_tensor(board)  # (12, 8, 8)
  board = BoardState.tensor_to_board(tensor)  # chess.Board


🔷 BƯỚC 2: self_play.py
─────────────────────────────────────
Purpose: Generate training data through self-play

Key classes:
  • SelfPlayGame
    - One game between two engines
    - Records all states and final result
    - play() → (result, reason)

  • SelfPlayManager
    - Multiple games management
    - play_games(num_games, white_mode, black_mode)
    - get_training_data_batch()
    - save_training_data(), load_training_data()

Example:
  manager = SelfPlayManager()
  stats = manager.play_games(num_games=20, 
                             white_mode='random',
                             black_mode='random')
  board_tensors, labels = manager.get_all_data()


🔷 BƯỚC 3: value_network.py
─────────────────────────────────────
Purpose: Neural network for position evaluation

Key classes:
  • ValueNetwork
    Architecture:
      Input (768) → Dense(128) + ReLU + Dropout
                  → Dense(64) + ReLU + Dropout
                  → Dense(1) + Tanh
      Output: value ∈ [-1, 1]

    Methods:
      • forward(x) → predictions
      • evaluate_position(board_tensor) → float
      • evaluate_positions_batch(batch) → numpy array

Example:
  network = ValueNetwork(hidden_size=128)
  value = network.evaluate_position(board_tensor)  # ≈0.45


🔷 BƯỚC 4: train.py
─────────────────────────────────────
Purpose: Train value network with PyTorch

Key classes:
  • ChessTrainer
    - train_epoch(dataloader) → loss
    - validate(dataloader) → loss
    - train(train_data, val_data, epochs)
    - save_model(), load_model()
    - plot_loss()

  • full_training_pipeline()
    - End-to-end: self-play → train → save

Example:
  trainer = ChessTrainer(network, learning_rate=0.001)
  result = trainer.train(train_data, val_data, epochs=100)


🔷 BƯỚC 5: minimax_engine.py
─────────────────────────────────────
Purpose: Minimax + Alpha-Beta + NN evaluation

Key classes:
  • MinimaxEngine
    - Minimax with alpha-beta pruning
    - Uses NN for terminal node evaluation
    - max_depth: search depth limit

    Methods:
      • get_best_move(board) → Move
      • get_best_move_with_score(board) → (Move, float)
      • minimax(board, depth, maximizing, α, β) → (value, move)

  • RandomEngine
    - Random move selector

Example:
  engine = MinimaxEngine(network, max_depth=3)
  move, score = engine.get_best_move_with_score(board)


🔷 BƯỚC 6: gui.py
─────────────────────────────────────
Purpose: Pygame GUI for interactive gameplay

Key classes:
  • ChessGUI
    - Draw 8×8 board
    - Draw pieces (unicode symbols)
    - Handle mouse clicks
    - AI automatic moves
    - Game status display

    Methods:
      • run() → main game loop
      • handle_click(pos, button)
      • ai_move()
      • reset()

Example:
  gui = create_gui_with_engine(model_path, ai_color=BLACK)
  gui.run()


🔷 main.py
─────────────────────────────────────
Purpose: Command-line interface entry point

Subcommands:
  • train
    - Full training pipeline
    - Usage: python main.py train --output-dir ./models

  • play
    - Interactive gameplay
    - Usage: python main.py play --depth 3 --player-color white

  • selfplay
    - Generate self-play data
    - Usage: python main.py selfplay --num-games 50 --white-mode minimax --save-data data.npz

  • analyze
    - Analyze position
    - Usage: python main.py analyze --depth 4 --fen "..."


# =============================================================================
# 🔄 WORKFLOW EXAMPLES
# =============================================================================

Workflow 1: FIRST TIME SETUP
───────────────────────────────
1. pip install -r requirements.txt
2. python main.py train
   (Creates self-play data + trains network)
3. python main.py play
   (Play vs AI)


Workflow 2: CONTINUOUS IMPROVEMENT
───────────────────────────────────
1. python main.py selfplay --num-games 100 \
     --white-mode minimax --black-mode minimax \
     --depth 3 --save-data data1.npz

2. python main.py train

3. python main.py selfplay --num-games 100 \
     --white-mode minimax --black-mode minimax \
     --depth 3 --save-data data2.npz
   
4. (Repeat 2-3)


Workflow 3: ANALYSIS ONLY
───────────────────────────────
1. python main.py analyze --depth 5
   (Analyze starting position)

2. python main.py analyze --depth 4 \
     --fen "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
   (Analyze specific position)


# =============================================================================
# 💾 DATA FORMATS
# =============================================================================

Board State:
  Type: numpy.ndarray
  Shape: (12, 8, 8)
  dtype: float32
  Range: [0, 1] (0=empty, 1=piece exists)

Label:
  Type: float32
  Range: [-1, 0, 1]
  -1 = Black wins (bad for White)
   0 = Draw
  +1 = White wins (good for White)

Training Data (.npz file):
  states: (N, 12, 8, 8) - board states
  labels: (N,) - game outcomes

Model (.pth file):
  PyTorch state_dict
  Loadable via: network.load_state_dict(...)


# =============================================================================
# 🎮 GAMEPLAY CONTROLS
# =============================================================================

While playing:
  • Left Click: Select piece or move to square
  • Right Click: Undo last move
  • R: Reset game
  • Q: Quit

Display shows:
  • 8×8 board with pieces
  • Highlighted selected square
  • Blue dots for legal moves
  • Right panel with:
    - Current player (Trắng/Đen)
    - Move history
    - Controls


# =============================================================================
# 📈 EXPECTED PROGRESSION
# =============================================================================

Training Loss:
  Epoch 1:     loss ≈ 0.85  (random)
  Epoch 10:    loss ≈ 0.45  (learning)
  Epoch 50:    loss ≈ 0.15  (progress)
  Epoch 100:   loss ≈ 0.02  (convergence)

Game Strength (vs Random):
  Untrained NN:     ~50% win
  After 10 epochs:  ~60% win
  After 50 epochs:  ~75% win
  After 100 epochs: ~85% win


# =============================================================================
# 🔍 DEBUGGING TIPS
# =============================================================================

Check board state:
  from board_state import BoardState
  tensor = BoardState.board_to_tensor(board)
  print(tensor.shape)  # Should be (12, 8, 8)

Check network output:
  value = network.evaluate_position(tensor)
  print(f"Value: {value:.4f}")  # Should be in [-1, 1]

Check minimax search:
  move, score = engine.get_best_move_with_score(board)
  print(f"Nodes: {engine.nodes_evaluated}")

Check training:
  print(f"Train loss: {trainer.train_losses}")
  print(f"Val loss: {trainer.val_losses}")

Inspect game:
  print(board)  # ASCII board
  print(board.fen())  # FEN notation
  print(board.move_stack)  # Move history


# =============================================================================
# 📚 KEY CONCEPTS
# =============================================================================

State Representation:
  ✓ 12 planes (6 white pieces + 6 black pieces)
  ✓ 8×8 binary matrix per plane
  ✓ Efficient for neural network processing

Self-play:
  ✓ AI plays against itself
  ✓ All states get same outcome label
  ✓ Creates diverse training data

Value Network:
  ✓ Takes board state as input
  ✓ Outputs evaluation score [-1, 1]
  ✓ Not a move generator (policy)

Minimax:
  ✓ Exhaustive game tree search
  ✓ Max/Min layers for W/B alternation
  ✓ Alpha-Beta pruning removes ~90% nodes

Alpha-Beta Pruning:
  ✓ Optimization of minimax
  ✓ Alpha: best value for maximizer
  ✓ Beta: best value for minimizer
  ✓ Cutoff when α ≥ β

Integration:
  ✓ Minimax finds moves (tactical)
  ✓ NN evaluates positions (strategic)
  ✓ Combination = strong AI


# =============================================================================
# 📝 NOTES & CAVEATS
# =============================================================================

• First run will be slow (untrained network)
  → Network will improve with training iterations

• GPU speedup is optional
  → CPU mode works fine for depth ≤ 3

• Board representation is not compact
  → Could use bitboards, but clarity is prioritized

• Self-play data has winner bias
  → All states in game get same label
  → Better methods exist (q-learning, etc.)

• No opening book or endgame tables
  → Could significantly improve opening/ending play

• Alpha-Beta pruning effectiveness varies
  → Depends on move ordering
  → Could be improved with killer moves


# =============================================================================
# 🚀 NEXT STEPS
# =============================================================================

1. Try all the examples in EXAMPLES.py
2. Run QUICKSTART.md commands
3. Read ARCHITECTURE.md for deep dive
4. Experiment with different depths/modes
5. Generate larger datasets and retrain
6. Modify network architecture and tune hyperparameters
7. Implement advanced techniques (policy head, MCTS, etc.)


# =============================================================================
# 🏆 PROJECT SUMMARY
# =============================================================================

✅ BƯỚC 1: State Representation (board_state.py)
   - Converts chess board to/from 12×8×8 tensor
   - Efficient neural network input format

✅ BƯỚC 2: Self-Play (self_play.py)
   - AI plays against itself repeatedly
   - Generates labeled training data

✅ BƯỚC 3: Neural Network (value_network.py)
   - 3-layer fully connected network
   - Evaluates position value [-1, 1]

✅ BƯỚC 4: Training (train.py)
   - PyTorch training loop with validation
   - MSE loss, Adam optimizer, early stopping

✅ BƯỚC 5: Minimax Integration (minimax_engine.py)
   - Minimax with alpha-beta pruning
   - NN evaluates leaf nodes

✅ BƯỚC 6: GUI (gui.py)
   - Interactive Pygame interface
   - Click-to-move gameplay

✅ Entry Point (main.py)
   - CLI with 4 subcommands
   - Easy one-line usage

✅ Documentation
   - README.md: Overview
   - QUICKSTART.md: Commands
   - ARCHITECTURE.md: Technical details
   - EXAMPLES.py: Code samples


# =============================================================================
# 🎉 READY TO USE!
# =============================================================================

Start here:
  1. python main.py train          (creates model)
  2. python main.py play           (play vs AI)
  3. python main.py analyze        (analyze positions)

Questions? Check:
  • README.md for overview
  • QUICKSTART.md for commands  
  • ARCHITECTURE.md for concepts
  • EXAMPLES.py for code samples

Enjoy your AI Chess! 🏁
"""
