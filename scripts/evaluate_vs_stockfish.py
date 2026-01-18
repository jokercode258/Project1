import argparse
import math
import time
import chess
import chess.pgn
import chess.engine
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

try:
    from trueskill import TrueSkill, Rating
    TRUESKILL_AVAILABLE = True
except ImportError:
    TRUESKILL_AVAILABLE = False
    print("TrueSkill not installed. Install with: pip install trueskill")

# Import project engines/networks
from chess_ai.engine.minimax_engine import MinimaxEngine
from chess_ai.network.value_network import ValueNetwork, ImprovedValueNetwork
import torch


def load_network_from_checkpoint(model_path: Optional[str], device: torch.device):
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        network_type = None
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                network_type = checkpoint.get('network_type', None)
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Detect network type from state_dict keys
        if network_type is None:
            if isinstance(state_dict, dict):
                has_conv = any(k.startswith('conv') for k in state_dict.keys())
                has_bn = any(k.startswith('bn') for k in state_dict.keys())
                if has_conv and has_bn:
                    network_type = 'ImprovedValueNetwork'
                else:
                    network_type = 'ValueNetwork'
            else:
                network_type = 'ValueNetwork'

        if network_type == 'ImprovedValueNetwork' or 'improved' in str(network_type).lower():
            net = ImprovedValueNetwork(dropout=0.3)
        else:
            net = ValueNetwork(hidden_size=128, dropout=0.3)

        net.load_state_dict(state_dict)
        net.to(device)
        print(f"Loaded network from {model_path} as {network_type}")
        return net
    else:
        # If no model, use untrained improved network as default
        net = ImprovedValueNetwork(dropout=0.3)
        net.to(device)
        print("No model provided; using untrained ImprovedValueNetwork")
        return net


def estimate_elo_vs_one(opponent_rating: float, wins: int, draws: int, losses: int) -> float:
    N = wins + draws + losses
    if N == 0:
        return float('nan')
    S = (wins + 0.5 * draws) / N
    if S <= 0.0:
        return float('-inf')
    if S >= 1.0:
        return float('inf')
    R_you = opponent_rating - 400 * math.log10((1.0 / S) - 1.0)
    return R_you


def run_match_with_trueskill(results: list, stockfish_elo: float = 3500.0):
    """
    Estimate Elo using TrueSkill for higher confidence.
    
    Args:
        results: list of (our_win, opp_win) tuples where our_win/opp_win in {True, False}
        stockfish_elo: reference Elo for Stockfish
    
    Returns:
        dict with 'elo', 'std_dev', 'mean_sigma'
    """
    if not TRUESKILL_AVAILABLE:
        return None
    
    ts = TrueSkill(mu=1500, sigma=200)  # Use Elo-scale: mu=1500, sigma~200
    
    # Initialize ratings
    sf_rating = Rating(mu=stockfish_elo, sigma=200)
    our_rating = Rating(mu=1500, sigma=200)
    
    # Play out all results
    for our_win, opp_win in results:
        if our_win and opp_win:
            # Draw: no change in TrueSkill (or update with 0.5 each)
            pass
        elif our_win:
            # We won
            our_rating, sf_rating = ts.rate_1vs1(our_rating, sf_rating)
        else:
            # We lost
            sf_rating, our_rating = ts.rate_1vs1(sf_rating, our_rating)
    
    # Final Elo estimate (in TrueSkill Elo scale)
    our_elo = our_rating.mu
    confidence = 1.0 / our_rating.sigma  # rough confidence metric (higher = more certain)
    
    return {
        'elo': our_elo,
        'mu': our_rating.mu,
        'sigma': our_rating.sigma,
        'std_dev': our_rating.sigma,
        'confidence': confidence,
    }


def run_match(stockfish_path: str,
              model_path: Optional[str],
              games: int = 100,
              per_move_time: float = 0.5,
              max_depth: int = 3,
              opponent_elo: float = 3500.0,
              stockfish_skill: int = 20,
              max_moves: int = 200,
              pgn_out: Optional[str] = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_network_from_checkpoint(model_path, device)
    net.eval()  # ← Set to evaluation mode (disable dropout)
    engine_py = MinimaxEngine(net, device=device, max_depth=max_depth)

    sf_path = Path(stockfish_path)
    if not sf_path.exists():
        raise FileNotFoundError(f"Stockfish binary not found: {stockfish_path}")

    # Open Stockfish with UCI options
    sf = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    
    # Configure Stockfish
    try:
        sf.configure({'Skill Level': int(stockfish_skill)})
        print(f"Stockfish configured with Skill Level = {stockfish_skill}")
    except Exception as e:
        print(f"Warning: Could not set Stockfish Skill Level: {e}")

    wins = 0
    draws = 0
    losses = 0
    games_summary = []
    trueskill_results = []  # for TrueSkill calculation

    pgn_file = open(pgn_out, 'w') if pgn_out else None

    for g in range(games):
        board = chess.Board()
        game = chess.pgn.Game()
        node = game

        # Alternate colors: even games our engine plays White
        our_white = (g % 2 == 0)

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                is_our_move = our_white
            else:
                is_our_move = not our_white

            if is_our_move:
                # Our engine (MinimaxEngine)
                best_move, _ = engine_py.get_best_move_with_score(board, time_limit=per_move_time)
                if best_move is None:
                    # Deterministic fallback: choose first legal move and continue
                    legal = list(board.legal_moves)
                    if not legal:
                        print("Warning: Engine returned no move and no legal moves available; ending game")
                        break
                    fallback = legal[0]
                    print(f"Warning: Engine returned no move; falling back to first legal move {fallback}")
                    best_move = fallback
                # Validate move is legal in current position
                if best_move not in board.legal_moves:
                    print(f"Warning: Best move {best_move} is not legal in position {board.fen()} - falling back to first legal move")
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    fallback = legal[0]
                    board.push(fallback)
                    node = node.add_variation(fallback)
                else:
                    board.push(best_move)
                    node = node.add_variation(best_move)
                move_count += 1
            else:
                # Stockfish move
                try:
                    res = sf.play(board, chess.engine.Limit(time=per_move_time))
                    if res.move is None:
                        break
                    board.push(res.move)
                    node = node.add_variation(res.move)
                    move_count += 1
                except Exception as e:
                    print(f"Stockfish error: {e}")
                    break

        # If loop exited due to move limit, board.is_game_over() may still be False
        result = board.result()  # '1-0', '0-1', '1/2-1/2' or '*' for unfinished
        game.headers['Result'] = result
        games_summary.append((g+1, result, our_white))

        # tally from our engine perspective
        our_result = None  # 'win', 'loss', 'draw'
        if result == '1-0':  # White (1) vs Black (0)
            if our_white:
                wins += 1
                our_result = 'win'
                trueskill_results.append((True, False))  # (our_win, opp_win)
            else:
                losses += 1
                our_result = 'loss'
                trueskill_results.append((False, True))
        elif result == '0-1':  # Black (1) vs White (0)
            if our_white:
                losses += 1
                our_result = 'loss'
                trueskill_results.append((False, True))
            else:
                wins += 1
                our_result = 'win'
                trueskill_results.append((True, False))
        elif result == '1/2-1/2':
            draws += 1
            our_result = 'draw'
            trueskill_results.append((True, True))  # draw: both get 0.5
        else:  # result == '*', unfinished -> adjudicate
            print("⚠️ Game unfinished ('*'). Adjudicating by material/NN evaluation...")
            # Adjudicate using material evaluation; if strong advantage, assign win/loss, else draw
            try:
                # Prefer material-only adjudication to avoid NN influence
                mat = engine_py._evaluate_material(board)
            except Exception:
                mat = 0.0
            # mat is in [-1,1] where positive favors White
            if mat > 0.3:
                # White is ahead -> if our engine was White, count win, else loss
                if our_white:
                    wins += 1
                    our_result = 'win'
                    trueskill_results.append((True, False))
                else:
                    losses += 1
                    our_result = 'loss'
                    trueskill_results.append((False, True))
            elif mat < -0.3:
                if our_white:
                    losses += 1
                    our_result = 'loss'
                    trueskill_results.append((False, True))
                else:
                    wins += 1
                    our_result = 'win'
                    trueskill_results.append((True, False))
            else:
                draws += 1
                our_result = 'draw'
                trueskill_results.append((True, True))

        if pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)
            pgn_file.write('\n')

        # quick progress with more detail
        print(f"Game {g+1}/{games} result: {result} (our={'W' if our_white else 'B'}) -> {our_result}")

    if pgn_file:
        pgn_file.close()

    sf.quit()

    print('\n' + '='*60)
    print('MATCH SUMMARY')
    print('='*60)
    print(f'Wins: {wins}, Draws: {draws}, Losses: {losses}, Total Games: {games}')

    # Simple Elo estimate
    est_elo = estimate_elo_vs_one(opponent_elo, wins, draws, losses)
    print(f'\n[Simple Method]')
    print(f'Estimated Elo vs opponent (opponent_elo={opponent_elo}): {est_elo:.1f}')

    # TrueSkill estimate (if available)
    if TRUESKILL_AVAILABLE:
        ts_result = run_match_with_trueskill(trueskill_results, opponent_elo)
        if ts_result:
            print(f'\n[TrueSkill Method - Higher Confidence]')
            print(f'Estimated Elo: {ts_result["elo"]:.1f}')
            print(f'Mu (rating mean): {ts_result["mu"]:.2f}')
            print(f'Sigma (uncertainty ±): {ts_result["sigma"]:.2f}')
            print(f'Confidence level: {ts_result["confidence"]:.2f}')

    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'games': games,
        'estimated_elo': est_elo,
        'games_summary': games_summary,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Minimax AI vs Stockfish and estimate Elo with TrueSkill')
    parser.add_argument('--stockfish', required=True, help='Path to Stockfish binary')
    parser.add_argument('--model', help='Path to trained model (pth) for Minimax engine', default=None)
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--time', type=float, default=0.5, help='Seconds per move for both engines')
    parser.add_argument('--depth', type=int, default=3, help='Max depth for Minimax engine')
    parser.add_argument('--opponent-elo', type=float, default=3500.0, help='Assumed Elo of Stockfish opponent')
    parser.add_argument('--skill', type=int, default=20, help='Stockfish Skill Level (0-20, default 20 = full strength)')
    parser.add_argument('--pgn-out', type=str, default=None, help='Optional PGN output file to save games')

    args = parser.parse_args()

    start = time.time()
    res = run_match(stockfish_path=args.stockfish,
                    model_path=args.model,
                    games=args.games,
                    per_move_time=args.time,
                    max_depth=args.depth,
                    opponent_elo=args.opponent_elo,
                    stockfish_skill=args.skill,
                    pgn_out=args.pgn_out)
    elapsed = time.time() - start
    print(f'\nElapsed time: {elapsed:.1f}s')
