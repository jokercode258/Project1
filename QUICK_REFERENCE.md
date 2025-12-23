"""
═════════════════════════════════════════════════════════════════════════════════
                    AI CHESS - QUICK REFERENCE CARD
═════════════════════════════════════════════════════════════════════════════════

📋 COMMAND REFERENCE
─────────────────────────────────────────────────────────────────────────────────

INSTALLATION:
  pip install -r requirements.txt
  python validate.py

TRAINING:
  python main.py train                          # Full pipeline
  python main.py train --output-dir ./models    # Custom output

PLAYING:
  python main.py play                           # Default settings
  python main.py play --depth 3                 # Set depth
  python main.py play --player-color white      # Choose color
  python main.py play --model ./models/chess_value_network.pth  # Load model

SELF-PLAY:
  python main.py selfplay --num-games 50       # Generate data
  python main.py selfplay --white-mode minimax --black-mode minimax --depth 3
  python main.py selfplay --save-data data.npz  # Save training data

ANALYSIS:
  python main.py analyze                        # Analyze start position
  python main.py analyze --depth 4              # Custom depth
  python main.py analyze --fen "..."            # Analyze FEN position


📦 FILE MAP
─────────────────────────────────────────────────────────────────────────────────

CORE MODULES:
  board_state.py        → 12×8×8 tensor representation
  value_network.py      → Neural network (768→128→64→1)
  minimax_engine.py     → Minimax + Alpha-Beta + NN evaluation
  self_play.py          → Self-play games & data generation
  train.py              → PyTorch training loop
  gui.py                → Pygame interactive GUI
  main.py               → CLI entry point

DOCUMENTATION:
  README.md             → Start here! Overview & quick start
  QUICKSTART.md         → Commands & workflows
  ARCHITECTURE.md       → Technical deep dive (600+ lines)
  EXAMPLES.py           → Code examples for each module
  INDEX.md              → Complete index & reference
  IMPLEMENTATION_SUMMARY.md → Project summary

CONFIG:
  requirements.txt      → Dependencies
  validate.py           → System validation


🎯 KEY CLASSES & FUNCTIONS
─────────────────────────────────────────────────────────────────────────────────

BoardState (board_state.py):
  board_to_tensor(board) → numpy (12, 8, 8)
  tensor_to_board(tensor) → chess.Board
  get_legal_moves_tensor(board) → mask
  get_game_result(board) → float [-1, 0, 1]

ValueNetwork (value_network.py):
  __init__(hidden_size=128, dropout=0.3)
  forward(x) → predictions
  evaluate_position(board_tensor) → float
  evaluate_positions_batch(batch) → numpy array

MinimaxEngine (minimax_engine.py):
  __init__(network, device, max_depth=3)
  get_best_move(board) → Move
  get_best_move_with_score(board) → (Move, float)
  minimax(board, depth, maximizing, α, β) → (value, move)

SelfPlayGame (self_play.py):
  play() → (result, reason)
  get_training_data() → [(state, label), ...]

SelfPlayManager (self_play.py):
  play_games(num_games, white_mode, black_mode) → stats
  get_all_data() → (board_tensors, labels)
  save_training_data(filepath)
  load_training_data(filepath)

ChessTrainer (train.py):
  train_epoch(dataloader) → loss
  validate(dataloader) → loss
  train(train_data, val_data, epochs) → result
  save_model(filepath)
  load_model(filepath)
  plot_loss(save_path)

ChessGUI (gui.py):
  run() → game loop
  handle_click(pos, button)
  ai_move()
  reset()


📊 HYPERPARAMETERS
─────────────────────────────────────────────────────────────────────────────────

Network:
  hidden_size: 128          # First hidden layer
  dropout: 0.3              # Regularization
  output_range: [-1, 1]     # Tanh activation

Training:
  learning_rate: 0.001      # Adam optimizer
  weight_decay: 1e-5        # L2 regularization
  batch_size: 32
  epochs: 100
  early_stopping: 10        # Patience

Minimax:
  max_depth: 3              # Search depth (1-5 typical)
  alpha_init: -inf
  beta_init: +inf

Self-play:
  num_games: 20-100
  max_moves_per_game: 50-100


🧩 DATA FORMATS
─────────────────────────────────────────────────────────────────────────────────

Board State (tensor):
  Shape: (12, 8, 8)
  dtype: float32
  Range: [0, 1]
  Planes: 0-5 white, 6-11 black

Label:
  Type: float32
  Value: -1.0 (black wins), 0.0 (draw), 1.0 (white wins)

Training Data (.npz):
  states: (N, 12, 8, 8)
  labels: (N,)

Model (.pth):
  PyTorch state_dict
  Loadable via: load_state_dict(torch.load(...))


⚡ PERFORMANCE TIPS
─────────────────────────────────────────────────────────────────────────────────

Make AI Stronger:
  • Increase depth: --depth 5 (but slower)
  • Train longer: epochs=200 (needs more data)
  • Use minimax: --white-mode minimax (stronger than random)

Make AI Faster:
  • Decrease depth: --depth 2 (weaker)
  • Use GPU: automatic if available
  • Reduce batch size: batch_size=16

Generate Better Data:
  • Use minimax vs minimax (best data)
  • Increase num_games: 100+ games
  • Vary depths: mix different depths

Improve Network:
  • Increase hidden size: 256 or 512
  • Add more layers: add Dense(128) → Dense(64)
  • Longer training: 200+ epochs
  • More data: 10k+ self-play games


🎮 GAMEPLAY TIPS
─────────────────────────────────────────────────────────────────────────────────

For beginners:
  • Start with depth 2
  • Play as white (easier)
  • Study AI's moves

For intermediate:
  • Use depth 3-4
  • Play both colors
  • Use analysis mode

For advanced:
  • High depth (5+) takes time
  • Batch training: 100+ games per iteration
  • Analyze lost positions


🐛 TROUBLESHOOTING
─────────────────────────────────────────────────────────────────────────────────

Problem: "ModuleNotFoundError"
Solution: pip install -r requirements.txt

Problem: "GPU out of memory"
Solution: Reduce batch_size or use device='cpu'

Problem: "AI takes too long"
Solution: Reduce depth or use smaller model

Problem: "Model doesn't improve"
Solution: Generate more self-play data first

Problem: "Pygame won't display"
Solution: Check X11 settings or use headless

Problem: "Board state shape error"
Solution: Verify input is chess.Board object

Problem: "Training loss stays high"
Solution: Check data quality, increase learning_rate


📈 EXPECTED RESULTS
─────────────────────────────────────────────────────────────────────────────────

After 1 training run (20 self-play games):
  • Loss: 0.3-0.5
  • Time: 5-10 minutes
  • Strength: Weak but playable

After 100 epochs training:
  • Loss: 0.02-0.05
  • Win rate vs random: 70-80%
  • Strength: Decent
  • Time per move: 1-2 seconds (depth 3)

After 1000+ self-play games + training:
  • Loss: 0.01-0.02
  • Win rate vs random: 90%+
  • Strength: Strong
  • Can beat casual players


🔗 INTEGRATION POINTS
─────────────────────────────────────────────────────────────────────────────────

Custom game board:
  • Replace chess.Board with custom implementation
  • Keep BoardState interface same

Different network architecture:
  • Modify ValueNetwork.__init__()
  • Ensure output shape (batch, 1)

Alternative engine:
  • Implement same interface as MinimaxEngine
  • get_best_move(board) → Move

Different GUI:
  • Keep same interface
  • Modify draw methods

Custom training data:
  • Load with self_play.SelfPlayManager.load_training_data()
  • Ensure shape (N, 12, 8, 8) and labels (N,)


📚 KEY INSIGHTS
─────────────────────────────────────────────────────────────────────────────────

1. Minimax finds best moves, NN evaluates positions
2. Alpha-Beta pruning removes ~90% of nodes
3. Self-play creates unlimited training data
4. Early stopping prevents overfitting
5. Larger depths = stronger but slower
6. 12 planes better than single board matrix
7. Tanh output [-1, 1] matches game outcomes
8. GUI runs independently of training
9. All components are modular & replaceable
10. System scales from 1 to 1000+ games


🎓 LEARNING RESOURCES
─────────────────────────────────────────────────────────────────────────────────

In this project:
  • ARCHITECTURE.md: Technical foundations
  • EXAMPLES.py: Code usage patterns
  • Source code: Implementation details

External:
  • python-chess docs: Board & move handling
  • PyTorch docs: Neural networks
  • Pygame docs: GUI development
  • Wikipedia: Minimax, Alpha-Beta
  • Books: "AI: A Modern Approach"


✅ VALIDATION CHECKLIST
─────────────────────────────────────────────────────────────────────────────────

Before using:
  ☐ pip install -r requirements.txt
  ☐ python validate.py (all checks pass)
  ☐ Read README.md
  ☐ Run one example: python main.py train

For development:
  ☐ Understand all 6 steps
  ☐ Review ARCHITECTURE.md
  ☐ Test each module individually
  ☐ Read source code comments

For deployment:
  ☐ Train model adequately (100+ epochs)
  ☐ Validate with test games
  ☐ Document any customizations
  ☐ Version control setup


🚀 GETTING STARTED NOW
─────────────────────────────────────────────────────────────────────────────────

⏱️ 5 minutes:
   1. pip install -r requirements.txt
   2. python validate.py
   3. Read README.md

⏱️ 15 minutes:
   4. python main.py train
   5. python main.py play

⏱️ 1 hour:
   6. Explore source code
   7. Read ARCHITECTURE.md
   8. Try different parameters

⏱️ Full day:
   9. Deep dive into implementation
   10. Modify & experiment
   11. Generate custom data
   12. Train stronger model


═════════════════════════════════════════════════════════════════════════════════
                    Ready to dive in? Start here:
                    python main.py train
═════════════════════════════════════════════════════════════════════════════════
"""
