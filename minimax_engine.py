"""
BƯỚC 5: KẾT HỢP NEURAL NETWORK VỚI MINIMAX
Minimax tìm nước đi, Neural Network đánh giá thế cờ
"""

import numpy as np
import chess
from typing import Tuple, Optional
from board_state import BoardState
from value_network import ValueNetwork
import torch


class MinimaxEngine:
    """
    Minimax với Alpha-Beta Pruning
    Sử dụng Neural Network để đánh giá node lá
    
    Nguyên lý:
    - Minimax duyệt cây
    - NN đánh giá node lá
    - Không sinh nước đi
    """
    
    def __init__(self, network: ValueNetwork, device: torch.device = None, 
                 max_depth: int = 4):
        """
        Args:
            network: ValueNetwork để đánh giá vị trí
            device: torch device (CPU/GPU)
            max_depth: Độ sâu tìm kiếm tối đa
        """
        self.network = network
        self.device = device if device else torch.device('cpu')
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Đánh giá vị trí bằng NN
        
        Returns:
            float ∈ [-1, 1]
        """
        tensor = BoardState.board_to_tensor(board)
        return self.network.evaluate_position(tensor, self.device)
    
    def minimax(self, board: chess.Board, depth: int, maximizing: bool,
                alpha: float = -float('inf'), beta: float = float('inf')) -> Tuple[float, Optional[chess.Move]]:
        """
        Minimax với Alpha-Beta Pruning
        
        Args:
            board: Vị trí cờ hiện tại
            depth: Độ sâu còn lại
            maximizing: True = maximize (trắng), False = minimize (đen)
            alpha: Alpha cutoff value
            beta: Beta cutoff value
            
        Returns:
            (value, best_move)
        """
        # Terminal node: kết thúc game hoặc hết độ sâu
        if depth == 0 or board.is_game_over():
            # Lấy kết quả nếu game kết thúc
            result = BoardState.get_game_result(board)
            if result is not None:
                return result, None
            
            # Nếu chưa kết thúc, dùng NN đánh giá
            value = self.evaluate_position(board)
            self.nodes_evaluated += 1
            return value, None
        
        best_move = None
        
        if maximizing:
            # Trắng tìm maximize
            max_eval = -float('inf')
            
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval, best_move
        
        else:
            # Đen tìm minimize
            min_eval = float('inf')
            
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval, best_move
    
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Tìm nước đi tốt nhất
        
        Returns:
            chess.Move hoặc None nếu game kết thúc
        """
        if board.is_game_over():
            return None
        
        self.nodes_evaluated = 0
        
        is_maximizing = board.turn == chess.WHITE
        _, best_move = self.minimax(board, self.max_depth, is_maximizing)
        
        return best_move
    
    def get_best_move_with_score(self, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
        """
        Tìm nước đi tốt nhất cùng với điểm số đánh giá
        
        Returns:
            (best_move, score)
        """
        if board.is_game_over():
            return None, None
        
        self.nodes_evaluated = 0
        
        is_maximizing = board.turn == chess.WHITE
        score, best_move = self.minimax(board, self.max_depth, is_maximizing)
        
        return best_move, score
    
    def set_max_depth(self, depth: int):
        """Thay đổi độ sâu tìm kiếm"""
        self.max_depth = depth


class RandomEngine:
    """
    Random move selector (dùng cho self-play ban đầu)
    """
    
    @staticmethod
    def get_best_move(board: chess.Board) -> Optional[chess.Move]:
        """Chọn nước đi ngẫu nhiên"""
        moves = list(board.legal_moves)
        if not moves:
            return None
        return np.random.choice(moves)


class HybridEngine:
    """
    Hybrid engine: kết hợp Minimax + NN
    Có thể chuyển đổi giữa random, minimax, và neural network
    """
    
    def __init__(self, network: ValueNetwork = None, max_depth: int = 4, device: torch.device = None):
        self.network = network
        self.minimax_engine = MinimaxEngine(network, device, max_depth) if network else None
        self.device = device
    
    def get_best_move(self, board: chess.Board, mode: str = 'minimax') -> Optional[chess.Move]:
        """
        Args:
            board: Chess board
            mode: 'random', 'minimax', 'nn'
        """
        if mode == 'random':
            return RandomEngine.get_best_move(board)
        elif mode == 'minimax':
            if self.minimax_engine is None:
                raise ValueError("Network chưa được khởi tạo cho minimax")
            return self.minimax_engine.get_best_move(board)
        else:
            raise ValueError(f"Mode không hợp lệ: {mode}")


if __name__ == "__main__":
    from value_network import ValueNetwork
    
    # Khởi tạo network
    network = ValueNetwork(hidden_size=128)
    device = torch.device('cpu')
    
    # Khởi tạo engine
    engine = MinimaxEngine(network, device, max_depth=3)
    
    # Test với board mở đầu
    board = chess.Board()
    print(f"FEN: {board.fen()}")
    print(f"Trắng đi")
    
    # Tìm nước đi tốt nhất
    best_move, score = engine.get_best_move_with_score(board)
    print(f"\nNước đi tốt nhất: {best_move}")
    print(f"Điểm số: {score:.4f}")
    print(f"Số node đánh giá: {engine.nodes_evaluated}")
    
    # Thực hiện nước đi
    if best_move:
        board.push(best_move)
        print(f"\nSau nước đi: {best_move}")
        print(f"FEN: {board.fen()}")
        
        # Đen đi
        print(f"\nĐen đi")
        best_move_black, score_black = engine.get_best_move_with_score(board)
        print(f"Nước đi tốt nhất: {best_move_black}")
        print(f"Điểm số: {score_black:.4f}")
