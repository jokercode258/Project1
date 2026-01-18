import torch
import chess
import random
from pathlib import Path
from chess_ai.engine.minimax_engine import MinimaxEngine
from chess_ai.network.value_network import ImprovedValueNetwork


class RandomEngine:
    @staticmethod
    def get_best_move(board):
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None


def load_model(model_path: str):
    model = ImprovedValueNetwork(dropout=0.3)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle both checkpoint format and plain state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def play_game(white_engine, black_engine, max_moves=100):
    """Play one game, return result: 1 (white wins), 0 (draw), -1 (black wins)"""
    board = chess.Board()
    moves_made = 0
    
    while not board.is_game_over() and moves_made < max_moves:
        if board.turn == chess.WHITE:
            move = white_engine.get_best_move(board)
        else:
            move = black_engine.get_best_move(board)
        
        if move is None:
            break
        if move not in board.legal_moves:
            # Illegal move, should not happen - fallback to random
            move = random.choice(list(board.legal_moves))
        board.push(move)
        moves_made += 1
    
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner == chess.WHITE:
            return 1
        elif outcome.winner == chess.BLACK:
            return -1
        else:
            return 0
    else:
        return 0  # Draw by moves exceeded


def estimate_elo_simple(wins, draws, losses):
    """Estimate Elo rating from results"""
    total = wins + draws + losses
    if total == 0:
        return float('nan')
    
    score = (wins + 0.5 * draws) / total
    if score >= 0.99:
        return 800
    if score <= 0.01:
        return -800
    
    # Rough formula: Elo_diff ≈ 400 * log10(score / (1 - score))
    # Assuming opponent is 1200 Elo
    import math
    opponent_elo = 1200
    my_elo = opponent_elo - 400 * math.log10((1.0 / score) - 1.0)
    return my_elo


def run_evaluation():
    device = torch.device('cpu')
    model_path = 'data/models/pgn/improved_network.pth'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print("=" * 60)
    print("LOCKED MINIMAX ENGINE EVALUATION")
    print("=" * 60)
    print(f"\nModel: {Path(model_path).name}")
    
    net = load_model(model_path)
    ai_engine = MinimaxEngine(network=net, device=device, max_depth=3)
    random_engine = RandomEngine()

    print(f"AI depth: 3")
    print(f"Opponent: Random (baseline)")
    
    # AI as White
    print(f"\n{'=' * 60}")
    print("PHASE 1: AI as WHITE vs Random as BLACK")
    print(f"{'=' * 60}")
    
    ai_white_wins = 0
    ai_white_draws = 0
    ai_white_losses = 0
    
    for game_num in range(5):
        result = play_game(ai_engine, random_engine, max_moves=50)
        if result == 1:
            ai_white_wins += 1
            print(f"Game {game_num+1}: WHITE WINS")
        elif result == 0:
            ai_white_draws += 1
            print(f"Game {game_num+1}: DRAW")
        else:
            ai_white_losses += 1
            print(f"Game {game_num+1}: BLACK WINS")
    
    print(f"\nResults: {ai_white_wins}W {ai_white_draws}D {ai_white_losses}L")
    
    # AI as Black
    print(f"\n{'=' * 60}")
    print("PHASE 2: Random as WHITE vs AI as BLACK")
    print(f"{'=' * 60}")
    
    ai_black_wins = 0
    ai_black_draws = 0
    ai_black_losses = 0
    
    for game_num in range(5):
        result = play_game(random_engine, ai_engine, max_moves=50)
        if result == -1:
            ai_black_wins += 1
            print(f"Game {game_num+1}: BLACK WINS (AI)")
        elif result == 0:
            ai_black_draws += 1
            print(f"Game {game_num+1}: DRAW")
        else:
            ai_black_losses += 1
            print(f"Game {game_num+1}: WHITE WINS (AI loses)")
    
    print(f"\nResults: {ai_black_wins}W {ai_black_draws}D {ai_black_losses}L")
    
    # Overall
    total_wins = ai_white_wins + ai_black_wins
    total_draws = ai_white_draws + ai_black_draws
    total_losses = ai_white_losses + ai_black_losses
    
    print(f"\n{'=' * 60}")
    print("OVERALL RESULTS")
    print(f"{'=' * 60}")
    print(f"Total: {total_wins}W {total_draws}D {total_losses}L out of 10 games")
    print(f"Score: {100 * (total_wins + 0.5 * total_draws) / 10:.1f}%")
    
    elo = estimate_elo_simple(total_wins, total_draws, total_losses)
    print(f"Estimated Elo vs Random: {elo:.0f}")
    
    return True


if __name__ == '__main__':
    success = run_evaluation()
    exit(0 if success else 1)
