"""
KIẾN TRÚC CHI TIẾT - AI CHESS SYSTEM
Minimax + Neural Network + Self-play + GUI
"""

# =============================================================================
# ARCHITECTURE OVERVIEW
# =============================================================================

┌─────────────────────────────────────────────────────────────────┐
│                     AI CHESS SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐      ┌──────────────────┐                  │
│  │   Player/GUI    │      │   Self-Play      │                  │
│  │   (Pygame)      │      │   Generator      │                  │
│  └────────┬────────┘      └────────┬─────────┘                  │
│           │                        │                             │
│           │                        ├─→ Training Data (state, label)
│           │                        │                             │
│           └────────────┬───────────┘                             │
│                        ↓                                          │
│            ┌──────────────────────┐                              │
│            │   Chess Board        │                              │
│            │  (python-chess)      │                              │
│            └──────────┬───────────┘                              │
│                       ↓                                           │
│      ┌────────────────────────────────┐                         │
│      │    Minimax Engine              │                         │
│      │  + Alpha-Beta Pruning          │                         │
│      ├────────────────────────────────┤                         │
│      │ • Find legal moves             │                         │
│      │ • Build game tree              │                         │
│      │ • Alpha-Beta cutoff            │                         │
│      │ • Return best move             │                         │
│      └────────────┬────────────────────┘                         │
│                   ↓ evaluate(leaf_node)                          │
│      ┌──────────────────────────┐                                │
│      │   Value Network (NN)     │                                │
│      ├──────────────────────────┤                                │
│      │ Input: 12x8x8 (board)    │                                │
│      │ Hidden: 128              │                                │
│      │ Hidden: 64               │                                │
│      │ Output: [-1, 1] (value)  │                                │
│      └──────────┬───────────────┘                                │
│                 ↑                                                 │
│      ┌──────────────────────────┐                                │
│      │   State Representation   │                                │
│      ├──────────────────────────┤                                │
│      │ 12 planes x 8x8:         │                                │
│      │ • Plane 0-5: White pieces│                                │
│      │ • Plane 6-11: Black pieces                                │
│      │ • Binary: 1=exists, 0=no │                                │
│      └──────────────────────────┘                                │
│                                                                   │
│  ┌──────────────────────────┐                                    │
│  │   Training Pipeline      │                                    │
│  ├──────────────────────────┤                                    │
│  │ DataLoader (batch)       │                                    │
│  │ ↓ Forward pass           │                                    │
│  │ ↓ MSE Loss               │                                    │
│  │ ↓ Backward (Adam)        │                                    │
│  │ ↓ Weight update          │                                    │
│  └──────────────────────────┘                                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘


# =============================================================================
# BƯỚC 1: STATE REPRESENTATION (board_state.py)
# =============================================================================

Input: chess.Board
Output: numpy array (12, 8, 8)

┌─────────────────────────────────────────────────────────────────┐
│             Board Representation                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Plane 0: White Pawns         Plane 6: Black Pawns              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │ . . . . . . . .         │  │ . . . . . . . .         │       │
│  │ . . . . . . . .         │  │ p p p p p p p p         │       │
│  │ . . . . . . . .         │  │ . . . . . . . .         │       │
│  │ . . . . . . . .         │  │ . . . . . . . .         │       │
│  │ . . . . . . . .         │  │ . . . . . . . .         │       │
│  │ . . . . . . . .         │  │ . . . . . . . .         │       │
│  │ P P P P P P P P         │  │ . . . . . . . .         │       │
│  │ . . . . . . . .         │  │ . . . . . . . .         │       │
│  └─────────────────────────┘  └─────────────────────────┘       │
│                                                                   │
│  Similarly for: N(1), B(2), R(3), Q(4), K(5) and n-k(7-11)      │
│                                                                   │
│  ┌─────────────────────────┐                                     │
│  │  Concatenate into       │                                     │
│  │  Shape: (12, 8, 8)      │                                     │
│  │                         │                                     │
│  │  Memory: 12*8*8*4 = 3KB │                                     │
│  │  Processing: ⚡ Fast!    │                                     │
│  └─────────────────────────┘                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

Pros:
✓ Planes are feature-rich and interpretable
✓ Efficient for CNN processing
✓ No information loss
✓ Standard representation for DL

Cons:
✗ Not compact (could use bits)
✗ But acceptable tradeoff for clarity


# =============================================================================
# BƯỚC 2: SELF-PLAY (self_play.py)
# =============================================================================

Purpose: Generate training data

Process:
┌──────────────────────────────────┐
│ Initialize board                 │
├──────────────────────────────────┤
│ Loop until game ends:            │
│   1. Record state (12x8x8)       │
│   2. Choose move (random/minimax)│
│   3. Push move to board          │
│   4. Check if game over          │
├──────────────────────────────────┤
│ Get result (win/draw/loss)       │
│ Assign label to all states       │
├──────────────────────────────────┤
│ Return: [(state, label), ...]    │
└──────────────────────────────────┘

Example:
Game 1: Random vs Random → Draw
  State 1 → Label = 0.0
  State 2 → Label = 0.0
  ...
  State 42 → Label = 0.0
  
Game 2: Minimax vs Random → Minimax wins
  State 1 → Label = 1.0 (nếu white = minimax)
  State 2 → Label = 1.0
  ...
  
Training data: List of 1000+ (board, result) pairs


# =============================================================================
# BƯỚC 3: NEURAL NETWORK (value_network.py)
# =============================================================================

Architecture:

Input Layer (12×8×8 = 768 neurons)
    │
    ├─→ Flatten
    │
    ├─→ Dense(768 → 128) + ReLU + Dropout(0.3)
    │
    ├─→ Dense(128 → 64) + ReLU + Dropout(0.3)
    │
    ├─→ Dense(64 → 1) + Tanh
    │
Output (1 neuron, value ∈ [-1, 1])

Matrix dimensions:
Input:    (batch, 768)
Layer 1:  (batch, 128)  → Parameters: 768×128 + 128 = 98,432
Layer 2:  (batch, 64)   → Parameters: 128×64 + 64 = 8,256
Layer 3:  (batch, 1)    → Parameters: 64×1 + 1 = 65

Total params: ≈106,753 (lightweight!)

Forward pass: ≈1ms per board on CPU


# =============================================================================
# BƯỚC 4: TRAINING (train.py)
# =============================================================================

Training Loop:

for epoch in range(epochs):
    
    for batch in train_loader:
        # Forward pass
        predictions = network(boards)          # (batch, 1)
        
        # Compute loss
        loss = MSELoss(predictions, labels)    # scalar
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss = validate(val_loader)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break

Expected progression:
Epoch 1   : loss = 0.85  (random guess)
Epoch 10  : loss = 0.45  (learning patterns)
Epoch 50  : loss = 0.15  (understanding positions)
Epoch 100 : loss = 0.02  (refined evaluation)


# =============================================================================
# BƯỚC 5: MINIMAX + ALPHA-BETA (minimax_engine.py)
# =============================================================================

Minimax Pseudocode:

function minimax(board, depth, maximizing, α, β):
    
    if depth == 0 or board.is_game_over():
        return evaluate(board), None
    
    if maximizing:  // White's turn
        value = -∞
        for move in legal_moves:
            board.push(move)
            score, _ = minimax(board, depth-1, False, α, β)
            board.pop()
            
            value = max(value, score)
            α = max(α, value)
            
            if β ≤ α:
                break  // Beta cutoff
        
        return value, best_move
    
    else:  // Black's turn
        value = +∞
        for move in legal_moves:
            board.push(move)
            score, _ = minimax(board, depth-1, True, α, β)
            board.pop()
            
            value = min(value, score)
            β = min(β, value)
            
            if β ≤ α:
                break  // Alpha cutoff
        
        return value, best_move

Key points:
✓ Traverses game tree
✓ Alpha-Beta pruning removes ~90% of nodes
✓ NN evaluates leaf nodes (terminal or depth limit)
✓ Returns best move + score


# =============================================================================
# BƯỚC 6: GUI (gui.py)
# =============================================================================

Game Loop:

while running:
    
    # Event handling
    for event in pygame.event.get():
        if MOUSE_CLICK:
            handle_click(event.pos)
    
    # AI move
    if board.turn == ai_color:
        move = engine.get_best_move(board)
        board.push(move)
    
    # Draw
    screen.fill(BG_COLOR)
    draw_board()          # 8x8 squares
    draw_pieces()         # Unicode symbols
    draw_highlights()     # Selected square + legal moves
    draw_info_panel()     # Status, history, controls
    pygame.display.flip()
    
    clock.tick(60)  # 60 FPS


Interactions:
┌──────────────────────────────────┐
│  Player Click                     │
├──────────────────────────────────┤
│  select_square = click_pos        │
│  legal_moves = get_legal_moves()  │
│  highlight_moves()                │
│                                   │
│  Player Click (destination)       │
├──────────────────────────────────┤
│  move = Move(from, to)            │
│  board.push(move)                 │
│  update_display()                 │
│                                   │
│  → AI thinks (Minimax search)     │
│  → AI makes move                  │
│  → Display updates                │
│  → Player's turn again            │
└──────────────────────────────────┘


# =============================================================================
# DATA FLOW
# =============================================================================

1. TRAINING PHASE
   ┌─────────────┐
   │ Self-play   │
   │ (20 games)  │
   └──────┬──────┘
          │
          ├─→ 1000+ training samples
          │   (board state, result)
          │
   ┌──────┴──────┐
   │ Train.py    │
   │ (100 epochs)│
   └──────┬──────┘
          │
          ├─→ Model weights saved
          │   chess_value_network.pth
          │
          └─→ Loss curve, checkpoints

2. PLAYING PHASE
   ┌──────────────┐
   │ Player clicks│
   └──────┬───────┘
          │
   ┌──────┴─────────┐
   │ GUI.py         │
   │ Show board,    │
   │ get input      │
   └──────┬─────────┘
          │
   ┌──────┴──────────────────┐
   │ Minimax engine           │
   │ • Traverse game tree     │
   │ • Alpha-beta pruning     │
   │ • Call NN at leaf nodes  │
   └──────┬──────────────────┘
          │
   ┌──────┴─────────────┐
   │ Value Network      │
   │ board → [0, 1]     │
   └──────┬─────────────┘
          │
          ├─→ Minimax returns
          │   best move
          │
   ┌──────┴─────────┐
   │ Execute move   │
   │ Update display │
   └────────────────┘


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

Components:
┌────────────────────┬─────────────┬──────────────┐
│ Component          │ Time (ms)   │ Bottleneck   │
├────────────────────┼─────────────┼──────────────┤
│ Board state to NN  │ 0.5         │ -            │
│ NN forward pass    │ 1-5         │ ◄ Slow on CPU│
│ Minimax search     │ 100-1000    │ ◄ Main time  │
│ GUI render         │ 5-10        │ -            │
├────────────────────┼─────────────┼──────────────┤
│ Total move (depth 3)           │ 500-1500ms   │
│ Total move (depth 4)           │ 1-5s         │
└────────────────────┴─────────────┴──────────────┘

Speedups:
✓ Alpha-Beta: 10-100x vs pure minimax
✓ GPU: 3-5x vs CPU
✓ Transposition table: 2-3x (future)
✓ Iterative deepening: allows anytime move (future)


# =============================================================================
# MEMORY USAGE
# =============================================================================

Board representation (1 board):
  12 planes × 8×8 × 4 bytes (float32) = 3 KB

Training dataset (1000 samples):
  1000 × 3 KB = 3 MB

Network parameters:
  ≈106,753 params × 4 bytes = 427 KB

Minimax search tree (depth 4):
  ≈30 million nodes × 1 KB = 30 GB (stored implicitly!)


# =============================================================================
# ALGORITHM COMPARISON
# =============================================================================

Method              Strength    Speed   Memory   Learning
────────────────────────────────────────────────────────
Random              ⭐          ⚡⚡⚡   ✓        ✓
Minimax (depth 2)   ⭐⭐        ⚡⚡    ✓        ✓
Minimax (depth 4)   ⭐⭐⭐      ⚡     ✓        ✓
NN only             ⭐⭐⭐      ⚡⚡⚡   ✓        ✓
NN + Minimax        ⭐⭐⭐⭐⭐   ⚡⚡   ✓        ✓


# =============================================================================
# TESTING STRATEGY
# =============================================================================

Unit Tests:
- board_state.py: Tensor shape, coordinate mapping
- value_network.py: Forward pass, gradient flow
- minimax_engine.py: Move legality, evaluation consistency
- self_play.py: Data format, result calculation

Integration Tests:
- Full training pipeline
- Play a full game
- Win/loss detection

Validation:
- Training loss decreases
- Validation loss plateaus (generalization)
- Win rate improves
- Game moves look reasonable


# =============================================================================
# FUTURE IMPROVEMENTS
# =============================================================================

Short term:
- Add opening book (standard chess openings)
- Implement killer moves heuristic
- Add transposition table

Medium term:
- Policy head (move probability)
- Endgame tablebase
- Self-play with rating system (Elo)

Long term:
- MCTS (Monte Carlo Tree Search)
- NNUE architecture (more efficient)
- Quantization for mobile
- Multi-GPU training


# =============================================================================
# REFERENCES
# =============================================================================

1. Minimax Algorithm
   https://en.wikipedia.org/wiki/Minimax

2. Alpha-Beta Pruning
   https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning

3. Neural Networks for Game Playing
   Deep Blue, AlphaGo, etc.

4. Python-Chess Library
   https://python-chess.readthedocs.io/

5. PyTorch
   https://pytorch.org/docs/

6. Pygame
   https://www.pygame.org/docs/
"""

# END ARCHITECTURE DOCUMENTATION
