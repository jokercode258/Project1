import argparse
import torch
from pathlib import Path
import logging

from training.self_play import full_training_pipeline, SelfPlayManager, ChessTrainer as SelfPlayTrainer
from training.pgn import train_from_pgn, ChessTrainer as PGNTrainer, ChesscomDownloader, download_chessdotcom_dataset
from chess_ai.network.value_network import ValueNetwork, ImprovedValueNetwork
from chess_ai.engine.minimax_engine import MinimaxEngine

from gui_module.gui import create_gui_with_engine, ChessGUI
import chess

ChessTrainer = PGNTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def train_command(args):
    """Train mode - self-play random"""
    print("=" * 60)
    print("TRAINING AI CHESS (SELF-PLAY RANDOM)")
    print("=" * 60 + "\n")
    
    # Output directory mặc định: ./data/models/self_play
    output_dir = args.output_dir if args.output_dir != './data' else './data/models/self_play'
    
    network, trainer = full_training_pipeline(output_dir=output_dir)
    print("\nTraining hoàn thành!")


def train_pgn_command(args):
    """Train mode - từ PGN files với Stockfish evaluation"""
    print("=" * 60)
    print("TRAINING AI CHESS (FROM PGN FILES)")
    print("=" * 60 + "\n")
    
    if not Path(args.pgn_source).exists():
        print(f"Lỗi: {args.pgn_source} không tồn tại")
        return
    
    # Output directory mặc định: ./data/models/pgn
    output_dir = args.output_dir if args.output_dir != './data' else './data/models/pgn'
    
    network, trainer = train_from_pgn(
        pgn_source=args.pgn_source,
        output_dir=output_dir,
        use_improved_network=args.improved_network,
        max_positions=args.max_positions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        stockfish_path=args.stockfish_path if hasattr(args, 'stockfish_path') and args.stockfish_path else None
    )
    
    if network and trainer:
        print("\n Training hoàn thành!")
    else:
        print("\n Training thất bại!")


def download_command(args):
    """Download PGN files từ Chess.com"""
    print("=" * 60)
    print("DOWNLOAD PGN FILES FROM CHESS.COM")
    print("=" * 60 + "\n")
    
    output_dir = args.output_dir
    
    if args.player:
        # Download từ một player cụ thể
        print(f"Downloading games from player: {args.player}\n")
        downloader = ChesscomDownloader(output_dir)
        filepath = downloader.download_player_games(args.player)
        
        if filepath:
            print(f"\n Downloaded: {filepath}")
            print(f"\nNext, run:")
            print(f"  python main.py train-pgn --pgn-source {output_dir} --improved-network")
        else:
            print(" Download failed")
    else:
        # Download từ top GMs
        print("Downloading from top Grandmasters...\n")
        files = download_chessdotcom_dataset(output_dir)
        
        if files:
            print(f"\n Downloaded {len(files)} files to {output_dir}")
            print(f"\nNext, run:")
            print(f"  python main.py train-pgn --pgn-source {output_dir} --improved-network")
        else:
            print(" Download failed")


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
    
    # Decide which model to use: prefer explicit args.model; otherwise try self_play model, then pgn model
    model_path = None
    if args.model and Path(args.model).exists():
        model_path = args.model
    else:
        # Try self_play model
        sp_candidate = Path('./data/models/self_play/best_model.pth')
        pgn_candidate = Path('./data/models/pgn/improved_network.pth')
        if sp_candidate.exists():
            model_path = str(sp_candidate)
            print(f"Using self-play model: {model_path}")
        elif pgn_candidate.exists():
            model_path = str(pgn_candidate)
            print(f"Using PGN-trained model: {model_path}")
        else:
            print("No pretrained model found; GUI will use untrained network.")

    # Tạo GUI
    gui = create_gui_with_engine(
        model_path=model_path,
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
    from training.self_play import SelfPlayManager
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
    
    print(f"\nStats:")
    print(f"  Trắng thắng: {stats['white_wins']}")
    print(f"  Đen thắng: {stats['black_wins']}")
    print(f"  Hòa: {stats['draws']}")
    print(f"  Win rate Trắng: {stats['white_winrate']:.2%}")
    print(f"  Training samples: {stats['training_samples']}")
    
    # Lưu training data
    if args.save_data:
        manager.save_training_data(args.save_data)
        print(f"Lưu training data: {args.save_data}")


def analyze_command(args):
    """Analyze mode"""
    print("=" * 60)
    print("ANALYZE POSITION")
    print("=" * 60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    network = ValueNetwork(hidden_size=128)
    if args.model and Path(args.model).exists():
        network.load_state_dict(torch.load(args.model, map_location=device))
    network.to(device)
    
    
    engine = MinimaxEngine(network, device=device, max_depth=args.depth)
    
    
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
        score, _ = engine.minimax(board, args.depth - 1, 
                                 maximizing=not board.turn)
        board.pop()
        moves_scores.append((score, move))
    
    moves_scores.sort(key=lambda x: x[0], reverse=True)
    for i, (score, move) in enumerate(moves_scores[:3], 1):
        print(f"  {i}. {move} (score={score:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='AI Chess: Minimax + Neural Network + Self-play + GUI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model (self-play random)')
    train_parser.add_argument('--output-dir', default='./data',
                            help='Output directory for models (default: ./data/models/self_play)')
    train_parser.set_defaults(func=train_command)
    
    # Train from PGN command
    train_pgn_parser = subparsers.add_parser('train-pgn', 
                                            help='Train from PGN files')
    train_pgn_parser.add_argument('--pgn-source', required=True,
                                help='Path to PGN file hoặc directory chứa PGN files')
    train_pgn_parser.add_argument('--output-dir', default='./data',
                                help='Output directory for models (default: ./data/models/pgn)')
    train_pgn_parser.add_argument('--improved-network', action='store_true',
                                help='Sử dụng ImprovedValueNetwork (CNN)')
    train_pgn_parser.add_argument('--max-positions', type=int, default=100000,
                                help='Max positions để extract từ PGN')
    train_pgn_parser.add_argument('--epochs', type=int, default=200,
                                help='Số epochs training')
    train_pgn_parser.add_argument('--batch-size', type=int, default=64,
                                help='Batch size')
    train_pgn_parser.add_argument('--patience', type=int, default=30,
                                help='Early stopping patience')
    train_pgn_parser.add_argument('--stockfish-path', default=None,
                                help='Đường dẫn tới Stockfish executable (nếu không có, sẽ tìm tự động)')
    train_pgn_parser.set_defaults(func=train_pgn_command)
    
    # Download command
    download_parser = subparsers.add_parser('download', 
                                           help='Download PGN files from Chess.com')
    download_parser.add_argument('--output-dir', default='./pgn_files',
                               help='Output directory for PGN files')
    download_parser.add_argument('--player', default=None,
                               help='Download from specific player (e.g., "nakamura")')
    download_parser.set_defaults(func=download_command)
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play vs AI')
    play_parser.add_argument('--model', default=None,
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
    analyze_parser.add_argument('--model', default='./data/models/pgn/improved_network.pth',
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
    