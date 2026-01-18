"""
PGN Parser - xử lý PGN files để tạo training data
Module này nằm trong training/pgn nhưng import từ chess_ai.board_state
"""

import os
import chess
import chess.pgn
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
import shutil
from pathlib import Path
import time

from chess_ai.board.board_state import BoardState

logger = logging.getLogger(__name__)


def find_stockfish_path() -> Optional[str]:
    # 1. Kiểm tra environment variable
    env_path = os.getenv('STOCKFISH_PATH')
    if env_path and os.path.exists(env_path):
        logger.info(f"Found Stockfish from STOCKFISH_PATH: {env_path}")
        return env_path
    
    # 2. Kiểm tra Windows common paths
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
    
    # 3. Tìm trong PATH
    stockfish_exe = shutil.which('stockfish') or shutil.which('stockfish.exe')
    if stockfish_exe:
        logger.info(f"Found Stockfish in PATH: {stockfish_exe}")
        return stockfish_exe
    
    logger.warning("Stockfish executable not found. Set STOCKFISH_PATH environment variable or ensure it's in PATH")
    return None


class PGNProcessor:
    """
    Xử lý PGN files để tạo training data
    """
    
    def __init__(self, stockfish_path: Optional[str] = None, 
                 min_depth: int = 20):
        """
        Args:
            stockfish_path: Đường dẫn đến Stockfish executable
                           Nếu None, sẽ cố gắng tìm trong system PATH
            min_depth: Độ sâu tối thiểu của game để dùng
        """
        self.stockfish_path = stockfish_path
        self.min_depth = min_depth
        self.stockfish = None
        
        # Cố gắng import stockfish
        try:
            from stockfish import Stockfish
            self._stockfish_class = Stockfish
            self._init_stockfish()
        except ImportError:
            logger.warning("Stockfish not installed. Install with: pip install stockfish")
            self._stockfish_class = None
    
    def _init_stockfish(self):
        """Khởi tạo Stockfish engine"""
        if self._stockfish_class is None:
            return
        # If no explicit path provided, try to locate the Stockfish binary first.
        if not self.stockfish_path:
            try:
                found = find_stockfish_path()
            except Exception:
                found = None
            if found:
                self.stockfish_path = found
            else:
                logger.warning("Stockfish executable not found; skipping Stockfish wrapper initialization")
                return

        try:
            # Determine target threads: prefer explicit environment variable,
            # otherwise use cpu_count() and cap at 8 by default
            try:
                threads = int(os.getenv('STOCKFISH_THREADS')) if os.getenv('STOCKFISH_THREADS') else None
            except Exception:
                threads = None

            if threads is None:
                    # During PGN training, use 1 thread (most stable, avoids crashes)
                    threads = 1

            # First attempt: pass parameters in constructor if supported
            params = {"Threads": threads} if threads else {}

            if self.stockfish_path:
                try:
                    # some versions accept a 'parameters' kwarg
                    self.stockfish = self._stockfish_class(path=self.stockfish_path, parameters=params) if params else self._stockfish_class(path=self.stockfish_path)
                except TypeError:
                    # fallback to simple constructor
                    self.stockfish = self._stockfish_class(path=self.stockfish_path)
            else:
                try:
                    self.stockfish = self._stockfish_class(parameters=params) if params else self._stockfish_class()
                except TypeError:
                    self.stockfish = self._stockfish_class()

            # Try to set threads after initialization using common method names
            try:
                if threads and hasattr(self.stockfish, 'update_engine_parameters'):
                    self.stockfish.update_engine_parameters({"Threads": threads})
                elif threads and hasattr(self.stockfish, 'set_engine_parameters'):
                    self.stockfish.set_engine_parameters({"Threads": threads})
                elif threads and hasattr(self.stockfish, 'set_parameters'):
                    self.stockfish.set_parameters({"Threads": threads})
                elif threads and hasattr(self.stockfish, 'set_option'):
                    # some wrappers expose a set_option(name, value)
                    try:
                        self.stockfish.set_option('Threads', threads)
                    except TypeError:
                        # maybe expects dict
                        self.stockfish.set_option({'Threads': threads})
            except Exception:
                logger.info("Unable to programmatically set Stockfish threads; continuing with default engine settings")

            logger.info(f"Stockfish initialized successfully (Threads={threads})")
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish: {e}")
            self.stockfish = None
    
    def _set_engine_fen(self, fen: str) -> bool:
        """Try several possible wrapper methods to set the engine position.
        Returns True if successful, False otherwise."""
        if not self.stockfish:
            return False

        # Common method names across wrappers
        candidates = [
            'set_fen_position',
            'set_fen',
            'set_position',
            'set_board_fen',
            'position',
        ]

        for name in candidates:
            if hasattr(self.stockfish, name):
                try:
                    method = getattr(self.stockfish, name)
                    # some methods expect a single fen string, others a list/moves
                    method(fen)
                    return True
                except TypeError:
                    # try passing as dict if wrapper expects different signature
                    try:
                        method({'fen': fen})
                        return True
                    except Exception:
                        continue
                except Exception:
                    continue

        return False

    def _get_engine_eval_obj(self):
        """Try to retrieve evaluation object from the wrapper using common method names."""
        if not self.stockfish:
            return None

        candidates = ['get_evaluation', 'evaluate', 'get_score']
        for name in candidates:
            if hasattr(self.stockfish, name):
                try:
                    method = getattr(self.stockfish, name)
                    return method()
                except Exception:
                    continue

        return None
    
    def parse_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse một PGN file và tạo training data
        
        Args:
            pgn_path: Đường dẫn đến PGN file
            max_games: Số game tối đa để parse (None = tất cả)
            
        Returns:
            (board_tensors, labels) - training data
        """
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
                    
                    # Check độ sâu của game (số nước đi)
                    depth = len(list(game.mainline_moves()))
                    if depth < self.min_depth:
                        continue
                    
                    # Tạo training data từ game này
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
        """
        Trích xuất training data từ một game
        
        Returns:
            (board_states, evaluations) hoặc None nếu không thành công
        """
        try:
            board = chess.Board()
            states = []
            evals = []
            
            for move in game.mainline_moves():
                # Lưu position trước khi move
                board_tensor = BoardState.board_to_tensor(board)
                states.append(board_tensor)
                
                # Evaluate vị trí
                if self.stockfish:
                    eval_score = self._evaluate_position(board)
                    evals.append(eval_score)
                else:
                    # Fallback: dùng result của game
                    result = self._get_game_result(game)
                    evals.append(result)
                
                board.push(move)
            
            return states, evals
        
        except Exception as e:
            logger.warning(f"Failed to extract game data: {e}")
            return None
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate vị trí sử dụng Stockfish
        Normalize score về [-1, 1]
        IMPORTANT: Always returns from WHITE's perspective
        
        Returns:
            float - evaluation score ∈ [-1, 1] from WHITE's perspective
        """
        if not self.stockfish:
            return 0.0

        # Try evaluation with one automatic restart & retry on engine crash
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            attempts += 1
            try:
                ok = self._set_engine_fen(board.fen())
                if not ok:
                    # If we couldn't set FEN, fallback to a simple query or raise
                    raise RuntimeError('Could not set FEN on Stockfish wrapper')

                eval_obj = self._get_engine_eval_obj()

                if eval_obj is None:
                    return 0.0

                # Normalize extraction of score from various wrapper return formats
                score = 0
                if isinstance(eval_obj, dict):
                    # stockfish pip returns {'type':'cp'|'mate', 'value': int}
                    if eval_obj.get('type') == 'mate':
                        mv = eval_obj.get('value', 0)
                        score = 30000 if mv > 0 else -30000
                    elif 'value' in eval_obj:
                        score = eval_obj['value']
                    elif 'score' in eval_obj:
                        score = eval_obj['score']
                    elif 'cp' in eval_obj:
                        score = eval_obj['cp']
                    else:
                        return 0.0
                else:
                    # object with attribute 'value' or numeric
                    if hasattr(eval_obj, 'value'):
                        score = getattr(eval_obj, 'value')
                    elif isinstance(eval_obj, (int, float)):
                        score = eval_obj
                    else:
                        return 0.0

                # Normalize centipawns to [-1,1] and handle extreme mate values
                # Use tanh(score / 100.0) for better label spread
                # Scaling factor: /200 was too aggressive (83.63% labels in [0.0, 0.2])
                # /100 gives 2x spread: ±50cp → tanh(0.5) ≈ 0.46, ±100cp → tanh(1.0) ≈ 0.76
                normalized = np.tanh(score / 100.0)
                
                # CRITICAL: Stockfish always evaluates from the perspective of the side to move
                # We need to convert to WHITE's perspective
                if not board.turn:  # If it's BLACK's turn to move, negate the score
                    normalized = -normalized
                
                return float(normalized)

            except Exception as e:
                logger.warning(f"Stockfish evaluation failed: {e}")
                # Attempt to reinitialize engine once if it crashed
                if attempts < max_attempts:
                    logger.info("Attempting to restart Stockfish engine and retry evaluation")
                    try:
                        # small pause to avoid rapid restart loops
                        time.sleep(0.1)
                        self._init_stockfish()
                    except Exception:
                        pass
                    continue
                return 0.0
    
    def _get_game_result(self, game) -> float:
        """
        Lấy kết quả game
        
        Returns:
            1.0 (Trắng thắng), 0.5 (Hòa), 0.0 (Đen thắng)
        """
        try:
            result = game.headers.get('Result', '*')
            if result == '1-0':  # Trắng thắng
                return 1.0
            elif result == '0-1':  # Đen thắng
                return 0.0
            elif result == '1/2-1/2':  # Hòa
                return 0.5
            else:
                return 0.5
        except:
            return 0.5
    
    def parse_pgn_directory(self, directory: str, max_positions: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse tất cả PGN files trong một directory
        
        Args:
            directory: Đường dẫn directory chứa PGN files
            max_positions: Tối đa số positions để extract
            
        Returns:
            (board_tensors, labels) - training data
        """
        all_tensors = []
        all_labels = []
        
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return np.array([]), np.array([])
        
        pgn_files = [f for f in os.listdir(directory) if f.endswith('.pgn')]
        logger.info(f"Found {len(pgn_files)} PGN files in {directory}")
        
        for pgn_file in pgn_files:
            pgn_path = os.path.join(directory, pgn_file)
            
            # Tính số positions còn cần
            remaining = max_positions - len(all_labels)
            if remaining <= 0:
                break
            
            logger.info(f"Processing {pgn_file}...")
            
            # Estimate số games cần (mỗi game ~ 40 positions)
            max_games = remaining // 40 + 1
            
            tensors, labels = self.parse_pgn_file(pgn_path, max_games=max_games)
            
            if len(tensors) > 0:
                all_tensors.extend(tensors)
                all_labels.extend(labels)
        
        logger.info(f"Total positions extracted: {len(all_labels)}")
        
        return np.array(all_tensors), np.array(all_labels)
    
    def save_training_data(self, board_tensors: np.ndarray, labels: np.ndarray, 
                          save_path: str = './data/datasets/pgn/pgn_training_data.npz'):
        """Lưu training data thành NPZ file"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, 
                     board_tensors=board_tensors,
                     labels=labels)
            logger.info(f"Training data saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
    
    def load_training_data(self, save_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data từ NPZ file"""
        try:
            data = np.load(save_path)
            return data['board_tensors'], data['labels']
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return np.array([]), np.array([])
