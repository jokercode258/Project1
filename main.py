"""
MAIN ENTRY POINT
"""

import argparse
import torch
from pathlib import Path

# Import modules
from train import full_training_pipeline, ChessTrainer
from value_network import ValueNetwork
from minimax_engine import MinimaxEngine
from self_play import SelfPlayManager
from gui import create_gui_with_engine, ChessGUI
import chess


def train_command(args):
    """Train mode"""
    print("=" * 60)
    print("TRAINING AI CHESS")
    print("=" * 60 + "\n")
    
    network, trainer = full_training_pipeline(output_dir=args.output_dir)
    print("\n✅ Training hoàn thành!")


def play_command(args):
    """Play mode"""
    print("=" * 60)
    print("PLAY CHESS vs AI")
    print("=" * 60 + "\n")
    
    # Color của người chơi
    player_color = chess.WHITE if args.player_color == 'white' else chess.BLACK
    ai_color = chess.BLACK if player_color == chess.WHITE else chess.WHITE
    
    print(f"Người chơi: {'Trắng' if player_color == chess.WHITE else 'Đen'}")
    print(f"AI: {'Trắng' if ai_color == chess.WHITE else 'Đen'}")
    print(f"Độ sâu Minimax: {args.depth}\n")
    
    # Tạo GUI
    gui = create_gui_with_engine(
        model_path=args.model,
        ai_color=ai_color,
        max_depth=args.depth
    )
    
    print("Bắt đầu game...")
    gui.run()


def selfplay_command(args):
    """Self-play mode"""
    print("=" * 60)
    print("SELF-PLAY")
    print("=" * 60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load network nếu có model
    network = None
    if args.model and Path(args.model).exists():
        network = ValueNetwork(hidden_size=128)
        network.load_state_dict(torch.load(args.model, map_location=device))
        network.to(device)
        print(f"Đã load model: {args.model}\n")
    
    # Self-play manager
    manager = SelfPlayManager(network=network, device=device, minimax_depth=args.depth)
    
    # Chơi games
    white_mode = args.white_mode
    black_mode = args.black_mode
    
    print(f"White: {white_mode}")
    print(f"Black: {black_mode}")
    print(f"Số game: {args.num_games}")
    print(f"Max moves: {args.max_moves}\n")
    
    stats = manager.play_games(
        num_games=args.num_games,
        white_mode=white_mode,
        black_mode=black_mode,
        max_moves=args.max_moves
    )
    
    print(f"\n📊 Stats:")
    print(f"  Trắng thắng: {stats['white_wins']}")
    print(f"  Đen thắng: {stats['black_wins']}")
    print(f"  Hòa: {stats['draws']}")
    print(f"  Win rate Trắng: {stats['white_winrate']:.2%}")
    print(f"  Training samples: {stats['training_samples']}")
    
    # Lưu training data
    if args.save_data:
        manager.save_training_data(args.save_data)
        print(f"✅ Lưu training data: {args.save_data}")


def analyze_command(args):
    """Analyze mode"""
    print("=" * 60)
    print("ANALYZE POSITION")
    print("=" * 60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load network
    network = ValueNetwork(hidden_size=128)
    if args.model and Path(args.model).exists():
        network.load_state_dict(torch.load(args.model, map_location=device))
    network.to(device)
    
    # Tạo engine
    engine = MinimaxEngine(network, device=device, max_depth=args.depth)
    
    # Tạo board
    board = chess.Board(args.fen) if args.fen else chess.Board()
    
    print(f"FEN: {board.fen()}\n")
    
    # Phân tích
    move, score = engine.get_best_move_with_score(board)
    
    print(f"Nước đi tốt nhất: {move}")
    print(f"Điểm số: {score:.4f}")
    print(f"Số node đánh giá: {engine.nodes_evaluated}")
    
    # Top 3 moves
    print(f"\nTop 3 nước đi:")
    moves_scores = []
    for move in list(board.legal_moves)[:10]:
        board.push(move)
        _, score = engine.minimax(board, args.depth - 1, 
                                 maximizing=not board.turn)
        board.pop()
        moves_scores.append((move, score))
    
    moves_scores.sort(key=lambda x: x[1], reverse=True)
    for i, (move, score) in enumerate(moves_scores[:3], 1):
        print(f"  {i}. {move} (score={score:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='AI Chess: Minimax + Neural Network + Self-play + GUI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--output-dir', default='./models',
                            help='Output directory for models')
    train_parser.set_defaults(func=train_command)
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play vs AI')
    play_parser.add_argument('--model', default='./models/chess_value_network.pth',
                           help='Path to model weights')
    play_parser.add_argument('--depth', type=int, default=3,
                           help='Minimax depth')
    play_parser.add_argument('--player-color', default='white',
                           choices=['white', 'black'],
                           help='Player color')
    play_parser.set_defaults(func=play_command)
    
    # Self-play command
    selfplay_parser = subparsers.add_parser('selfplay', help='Self-play games')
    selfplay_parser.add_argument('--num-games', type=int, default=20,
                               help='Number of games')
    selfplay_parser.add_argument('--white-mode', default='random',
                               choices=['random', 'minimax'],
                               help='White engine mode')
    selfplay_parser.add_argument('--black-mode', default='random',
                               choices=['random', 'minimax'],
                               help='Black engine mode')
    selfplay_parser.add_argument('--max-moves', type=int, default=100,
                               help='Max moves per game')
    selfplay_parser.add_argument('--model', default=None,
                               help='Path to model weights')
    selfplay_parser.add_argument('--depth', type=int, default=3,
                               help='Minimax depth')
    selfplay_parser.add_argument('--save-data', default=None,
                               help='Save training data to file')
    selfplay_parser.set_defaults(func=selfplay_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze position')
    analyze_parser.add_argument('--model', default='./models/chess_value_network.pth',
                              help='Path to model weights')
    analyze_parser.add_argument('--depth', type=int, default=3,
                              help='Minimax depth')
    analyze_parser.add_argument('--fen', default=None,
                              help='FEN string (default: starting position)')
    analyze_parser.set_defaults(func=analyze_command)
    
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
