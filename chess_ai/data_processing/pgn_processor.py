import os
import chess
import chess.pgn
import numpy as np
from typing import Tuple, List, Optional
from chess_ai.board.board_state import BoardState
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def find_stockfish_path() -> Optional[str]:
    env_path = os.getenv('STOCKFISH_PATH')
    if env_path and os.path.exists(env_path):
        logger.info(f"Found Stockfish from STOCKFISH_PATH: {env_path}")
        return env_path
    windows_paths = [
        r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\Program Files\Stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\Program Files (x86)\Stockfish\stockfish-windows-x86-64-avx2.exe",
        "stockfish-windows-x86-64-avx2.exe",
        "stockfish.exe",
    ]
    for path in windows_paths:
        if os.path.exists(path):
            logger.info(f"Found Stockfish at: {path}")
            return path
    stockfish_exe = shutil.which('stockfish') or shutil.which('stockfish.exe')
    if stockfish_exe:
        logger.info(f"Found Stockfish in PATH: {stockfish_exe}")
        return stockfish_exe
    logger.warning("Stockfish executable not found. Set STOCKFISH_PATH environment variable or ensure it's in PATH")
    return None


class PGNProcessor:
    def __init__(self, stockfish_path: Optional[str] = None, min_depth: int = 20):
        self.stockfish_path = stockfish_path
        self.min_depth = min_depth
        self.stockfish = None
        try:
            from stockfish import Stockfish
            self._stockfish_class = Stockfish
            self._init_stockfish()
        except ImportError:
            logger.warning("Stockfish not installed. Install with: pip install stockfish")
            self._stockfish_class = None

    def _init_stockfish(self):
        if self._stockfish_class is None:
            return
        try:
            if self.stockfish_path:
                self.stockfish = self._stockfish_class(path=self.stockfish_path)
            else:
                self.stockfish = self._stockfish_class()
            logger.info("Stockfish initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish: {e}")
            self.stockfish = None

    def parse_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        board_tensors = []
        labels = []
        if not os.path.exists(pgn_path):
            logger.error(f"PGN file not found: {pgn_path}")
            return np.array([]), np.array([])
        game_count = 0
        try:
            with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    if max_games and game_count >= max_games:
                        break
                    depth = len(list(game.mainline_moves()))
                    if depth < self.min_depth:
                        continue
                    game_data = self._extract_game_data(game)
                    if game_data:
                        states, evals = game_data
                        board_tensors.extend(states)
                        labels.extend(evals)
                    game_count += 1
                    if (game_count + 1) % 100 == 0:
                        logger.info(f"Processed {game_count + 1} games, {len(labels)} positions")
        except Exception as e:
            logger.error(f"Error parsing PGN file: {e}")
        logger.info(f"Total games processed: {game_count}")
        logger.info(f"Total positions extracted: {len(labels)}")
        return np.array(board_tensors), np.array(labels)

    def _extract_game_data(self, game) -> Optional[Tuple[List[np.ndarray], List[float]]]:
        try:
            board = chess.Board()
            states = []
            evals = []
            for move in game.mainline_moves():
                board_tensor = BoardState.board_to_tensor(board)
                states.append(board_tensor)
                if self.stockfish:
                    eval_score = self._evaluate_position(board)
                    evals.append(eval_score)
                else:
                    result = self._get_game_result(game)
                    evals.append(result)
                board.push(move)
            return states, evals
        except Exception as e:
            logger.warning(f"Failed to extract game data: {e}")
            return None

    def _evaluate_position(self, board: chess.Board) -> float:
        if not self.stockfish:
            return 0.0
        try:
            self.stockfish.set_fen_position(board.fen())
            eval_obj = self.stockfish.get_evaluation()
            if eval_obj is None:
                return 0.0
            if isinstance(eval_obj, dict):
                if 'value' in eval_obj:
                    score = eval_obj['value']
                else:
                    return 0.0
            else:
                score = eval_obj.value if hasattr(eval_obj, 'value') else 0
            normalized = np.tanh(score / 1000.0)
            return float(normalized)
        except Exception as e:
            logger.warning(f"Stockfish evaluation failed: {e}")
            return 0.0

    def _get_game_result(self, game) -> float:
        try:
            result = game.headers.get('Result', '*')
            if result == '1-0':
                return 1.0
            elif result == '0-1':
                return 0.0
            elif result == '1/2-1/2':
                return 0.5
            else:
                return 0.5
        except:
            return 0.5

    def parse_pgn_directory(self, directory: str, max_positions: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        all_tensors = []
        all_labels = []
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return np.array([]), np.array([])
        pgn_files = [f for f in os.listdir(directory) if f.endswith('.pgn')]
        logger.info(f"Found {len(pgn_files)} PGN files in {directory}")
        for pgn_file in pgn_files:
            pgn_path = os.path.join(directory, pgn_file)
            remaining = max_positions - len(all_labels)
            if remaining <= 0:
                break
            logger.info(f"Processing {pgn_file}...")
            max_games = remaining // 40 + 1
            tensors, labels = self.parse_pgn_file(pgn_path, max_games=max_games)
            if len(tensors) > 0:
                all_tensors.extend(tensors)
                all_labels.extend(labels)
        logger.info(f"Total positions extracted: {len(all_labels)}")
        return np.array(all_tensors), np.array(all_labels)

    def save_training_data(self, board_tensors: np.ndarray, labels: np.ndarray, save_path: str = 'pgn_training_data.npz'):
        try:
            np.savez(save_path, board_tensors=board_tensors, labels=labels)
            logger.info(f"Training data saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

    def load_training_data(self, save_path: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            data = np.load(save_path)
            return data['board_tensors'], data['labels']
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return np.array([]), np.array([])
