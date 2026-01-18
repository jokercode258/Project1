import numpy as np
import chess
from typing import List, Tuple
import torch

# Import từ chess_ai core
from chess_ai.board.board_state import BoardState
from chess_ai.engine.minimax_engine import RandomEngine, MinimaxEngine
from chess_ai.network.value_network import ValueNetwork


class SelfPlayGame:
    def __init__(self, white_engine = None, black_engine = None, max_moves: int = 100):
        """
        Args:
            white_engine: Engine cho Trắng (nếu None → dùng random)
            black_engine: Engine cho Đen (nếu None → dùng random)
            max_moves: Số nước đi tối đa
        """
        self.white_engine = white_engine
        self.black_engine = black_engine
        self.max_moves = max_moves
        
        self.board = chess.Board()
        self.game_data = []  # [(state, result), ...]
    
    def play(self) -> Tuple[int, str]:
        """
        Chơi một trận tự động
        
        Returns:
            (result, reason)
            result: 1 (Trắng thắng), 0 (Hòa), -1 (Đen thắng)
            reason: Lý do kết thúc ('checkmate', 'stalemate', 'max_moves', ...)
        """
        move_count = 0
        
        while move_count < self.max_moves and not self.board.is_game_over():
            # Lưu state trước khi chơi
            state = BoardState.board_to_tensor(self.board)
            self.game_data.append(state)
            
            # Chọn nước đi
            if self.board.turn == chess.WHITE:
                if self.white_engine:
                    move = self.white_engine.get_best_move(self.board)
                else:
                    move = RandomEngine.get_best_move(self.board)
            else:
                if self.black_engine:
                    move = self.black_engine.get_best_move(self.board)
                else:
                    move = RandomEngine.get_best_move(self.board)
            
            if move is None:
                break
            
            self.board.push(move)
            move_count += 1
        
        # Lấy kết quả
        result_obj = BoardState.get_game_result(self.board)
        
        if result_obj is None:
            # Game không kết thúc (max_moves)
            result = 0  # Coi như hòa
            reason = 'max_moves'
        else:
            result = int(result_obj)
            reason = self._get_end_reason()
        
        return result, reason
    
    def _get_end_reason(self) -> str:
        """Lý do kết thúc game"""
        outcome = self.board.outcome()
        if outcome.termination == chess.Termination.CHECKMATE:
            return 'checkmate'
        elif outcome.termination == chess.Termination.STALEMATE:
            return 'stalemate'
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return 'insufficient_material'
        elif outcome.termination == chess.Termination.FIFTYEIGHT_MOVE_RULE:
            return 'fifty_move_rule'
        else:
            return 'other'
    
    def get_training_data(self) -> List[Tuple[np.ndarray, float]]:
        """
        Lấy dữ liệu training: (board_tensor, result)
        
        Returns:
            List[(state, label)] với label = 1 (trắng thắng), 0 (hòa), -1 (đen thua)
        """
        result = BoardState.get_game_result(self.board)
        if result is None:
            result = 0  # Hòa
        
        training_data = []
        for state in self.game_data:
            training_data.append((state, result))
        
        return training_data


class SelfPlayManager:
    """
    Quản lý multiple self-play games để tạo dataset
    """
    
    def __init__(self, network: ValueNetwork = None, device: torch.device = None,
                 minimax_depth: int = 3):
        """
        Args:
            network: Neural network cho Minimax engine
            device: torch device
            minimax_depth: Độ sâu Minimax
        """
        self.network = network
        self.device = device if device else torch.device('cpu')
        self.minimax_depth = minimax_depth
        self.minimax_engine = None
        
        if network:
            self.minimax_engine = MinimaxEngine(network, device, minimax_depth)
        
        self.all_training_data = []
        self.game_stats = []
    
    def play_games(self, num_games: int = 10, white_mode: str = 'random', 
                   black_mode: str = 'random', max_moves: int = 100) -> dict:
        """
        Chơi multiple games
        
        Args:
            num_games: Số game
            white_mode: 'random' hoặc 'minimax'
            black_mode: 'random' hoặc 'minimax'
            max_moves: Số nước đi tối đa per game
            
        Returns:
            {'total_games': int, 'white_wins': int, 'black_wins': int, 'draws': int, ...}
        """
        # Khởi tạo engines
        white_engine = None
        if white_mode == 'minimax' and self.minimax_engine:
            white_engine = self.minimax_engine
        
        black_engine = None
        if black_mode == 'minimax' and self.minimax_engine:
            black_engine = self.minimax_engine
        
        # Chơi games
        white_wins = 0
        black_wins = 0
        draws = 0
        
        for game_num in range(num_games):
            game = SelfPlayGame(white_engine, black_engine, max_moves)
            result, reason = game.play()
            
            # Cộng stats
            if result == 1:
                white_wins += 1
            elif result == -1:
                black_wins += 1
            else:
                draws += 1
            
            # Lưu stats
            self.game_stats.append({
                'game': game_num,
                'result': result,
                'reason': reason,
                'moves': len(game.game_data),
                'white_mode': white_mode,
                'black_mode': black_mode
            })
            
            # Lưu training data
            training_data = game.get_training_data()
            self.all_training_data.extend(training_data)
            
            if (game_num + 1) % max(1, num_games // 10) == 0:
                print(f"Hoàn thành game {game_num + 1}/{num_games}")
        
        stats = {
            'total_games': num_games,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'white_winrate': white_wins / num_games,
            'training_samples': len(self.all_training_data)
        }
        
        return stats
    
    def get_training_data_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy batch training data
        
        Returns:
            (board_tensors, labels)
            board_tensors: (batch_size, 12, 8, 8)
            labels: (batch_size,)
        """
        if len(self.all_training_data) < batch_size:
            print(f"Cảnh báo: Chỉ có {len(self.all_training_data)} mẫu, cần {batch_size}")
        
        indices = np.random.choice(len(self.all_training_data), batch_size, replace=True)
        
        board_tensors = []
        labels = []
        
        for idx in indices:
            state, label = self.all_training_data[idx]
            board_tensors.append(state)
            labels.append(label)
        
        return np.array(board_tensors, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy tất cả training data
        
        Returns:
            (board_tensors, labels)
        """
        if not self.all_training_data:
            return np.array([]), np.array([])
        
        board_tensors = np.array([state for state, _ in self.all_training_data], dtype=np.float32)
        labels = np.array([label for _, label in self.all_training_data], dtype=np.float32)
        
        return board_tensors, labels
    
    def save_training_data(self, filepath: str):
        """Lưu training data"""
        board_tensors, labels = self.get_all_data()
        np.savez(filepath, states=board_tensors, labels=labels)
        print(f"Lưu {len(labels)} mẫu vào {filepath}")
    
    def load_training_data(self, filepath: str):
        """Load training data"""
        data = np.load(filepath)
        board_tensors = data['states']
        labels = data['labels']
        
        self.all_training_data = [(board_tensors[i], labels[i]) for i in range(len(labels))]
        print(f"Load {len(labels)} mẫu từ {filepath}")
