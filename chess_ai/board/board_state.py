import numpy as np
from typing import Tuple, List
import chess


class BoardState:
    """
    Chuyển bàn cờ chess -> tensor 12x8x8
    Plane 0-5: Quân trắng
    Plane 6-11: Quân đen
    """
    
    # Mapping loại quân -> index plane
    PIECE_TO_PLANE = {
        # Trắng (0-5)
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    @staticmethod
    def board_to_tensor(board: chess.Board) -> np.ndarray:
        """
        Chuyển chess.Board -> tensor (12, 8, 8)
        """
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            if piece is None:
                continue
            
            # Tính hàng và cột
            row = 7 - (square // 8)
            col = square % 8
            
            # Plane index = piece_type + (6 nếu đen)
            plane_idx = BoardState.PIECE_TO_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane_idx += 6
            
            tensor[plane_idx, row, col] = 1.0
        
        return tensor
    
    @staticmethod
    def tensor_to_board(tensor: np.ndarray) -> chess.Board:
        """
        Chuyển tensor (12, 8, 8) -> chess.Board
        """
        board = chess.Board(fen='8/8/8/8/8/8/8/8 w KQkq - 0 1')
        board.clear()
        
        for plane_idx in range(12):
            # Xác định loại quân và màu
            if plane_idx < 6:
                color = chess.WHITE
                piece_type = list(BoardState.PIECE_TO_PLANE.keys())[ 
                    list(BoardState.PIECE_TO_PLANE.values()).index(plane_idx)
                ]
            else:
                color = chess.BLACK
                piece_type = list(BoardState.PIECE_TO_PLANE.keys())[ 
                    list(BoardState.PIECE_TO_PLANE.values()).index(plane_idx - 6)
                ]
            
            piece = chess.Piece(piece_type, color)
            
            # Tìm tất cả ô có giá trị 1
            for row in range(8):
                for col in range(8):
                    if tensor[plane_idx, row, col] > 0.5:
                        # Tính square từ row, col
                        square = (7 - row) * 8 + col
                        board.set_piece_at(square, piece)
        
        return board
    
    @staticmethod
    def get_legal_moves_tensor(board: chess.Board) -> np.ndarray:
        """
        Tạo mask cho nước đi hợp lệ
        
        Returns:
            numpy array (64,) với 1 ở nước đi hợp lệ, 0 ở nước thua
        """
        mask = np.zeros(64, dtype=np.float32)
        
        for move in board.legal_moves:
            to_square = move.to_square
            mask[to_square] = 1.0
        
        return mask
    
    @staticmethod
    def board_to_fen(tensor: np.ndarray, is_white_turn: bool) -> str:
        """
        Chuyển tensor → FEN string
        """
        board = BoardState.tensor_to_board(tensor)
        board.turn = chess.WHITE if is_white_turn else chess.BLACK
        return board.fen()
    
    @staticmethod
    def get_game_result(board: chess.Board) -> int:
        """
        Lấy kết quả game cho Trắng

        Returns:
            1.0 nếu Trắng thắng
            0.0 nếu hòa
            -1.0 nếu Đen thắng (Trắng thua)
            None nếu game chưa kết thúc
        """
        if not board.is_game_over():
            return None
        
        result = board.outcome()
        
        if result.winner == chess.WHITE:
            return 1.0
        elif result.winner == chess.BLACK:
            return -1.0
        else:  # Hòa
            return 0.0


if __name__ == "__main__":
    # Test
    board = chess.Board()
    
    # Chuyển sang tensor
    tensor = BoardState.board_to_tensor(board)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Số quân trắng: {tensor[:6].sum()}")
    print(f"Số quân đen: {tensor[6:].sum()}")
    
    # Test inverse
    board_restored = BoardState.tensor_to_board(tensor)
    print(f"\nFEN gốc: {board.fen()}")
    print(f"FEN restore: {board_restored.fen()}")
    print(f"Bằng nhau: {board.fen() == board_restored.fen()}")
