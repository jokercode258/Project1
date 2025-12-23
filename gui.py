"""
BƯỚC 6: GIAO DIỆN PYGAME (CHƠI ĐƯỢC)
Cho người dùng chơi trực tiếp với AI
"""

import pygame
import chess
import numpy as np
from typing import Optional, Tuple
from board_state import BoardState
from minimax_engine import MinimaxEngine, RandomEngine
from value_network import ValueNetwork
import torch
from pathlib import Path


class ChessGUI:
    """
    Giao diện Pygame để chơi cờ
    """
    
    # Constants
    BOARD_SIZE = 800
    SQUARE_SIZE = BOARD_SIZE // 8
    
    # Colors
    WHITE_SQUARE = (240, 217, 181)
    BLACK_SQUARE = (181, 136, 99)
    HIGHLIGHT_COLOR = (186, 202, 44)
    MOVE_COLOR = (100, 150, 255)
    BG_COLOR = (50, 50, 50)
    TEXT_COLOR = (255, 255, 255)
    
    # Piece unicode
    PIECE_SYMBOLS = {
        chess.PAWN: '♟',
        chess.KNIGHT: '♞',
        chess.BISHOP: '♗',
        chess.ROOK: '♜',
        chess.QUEEN: '♛',
        chess.KING: '♚'
    }
    
    def __init__(self, engine: MinimaxEngine = None, ai_color: chess.Color = chess.BLACK,
                 ai_mode: str = 'minimax'):
        """
        Args:
            engine: AI engine (MinimaxEngine hoặc RandomEngine)
            ai_color: Màu AI (chess.WHITE hoặc chess.BLACK)
            ai_mode: 'minimax' hoặc 'random'
        """
        pygame.init()
        
        self.screen = pygame.display.set_mode((1200, 800))
        pygame.display.set_caption("AI Chess - Minimax + Neural Network")
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.board = chess.Board()
        self.engine = engine
        self.ai_color = ai_color
        self.ai_mode = ai_mode
        
        # Game state
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.game_result = None
        self.move_history = []
    
    def draw_board(self):
        """Vẽ bàn cờ 8x8"""
        for row in range(8):
            for col in range(8):
                x = col * self.SQUARE_SIZE + 50
                y = row * self.SQUARE_SIZE + 50
                
                # Vẽ hình vuông
                if (row + col) % 2 == 0:
                    color = self.WHITE_SQUARE
                else:
                    color = self.BLACK_SQUARE
                
                pygame.draw.rect(self.screen, color, 
                               (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                # Highlight nước đi có thể
                square = (7 - row) * 8 + col
                if square in [m.to_square for m in self.legal_moves]:
                    pygame.draw.circle(self.screen, self.MOVE_COLOR,
                                     (x + self.SQUARE_SIZE // 2, 
                                      y + self.SQUARE_SIZE // 2), 8)
        
        # Highlight selected square
        if self.selected_square is not None:
            row = 7 - (self.selected_square // 8)
            col = self.selected_square % 8
            x = col * self.SQUARE_SIZE + 50
            y = row * self.SQUARE_SIZE + 50
            pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR,
                           (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 3)
    
    def draw_pieces(self):
        """Vẽ các quân cờ"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            
            row = 7 - (square // 8)
            col = square % 8
            x = col * self.SQUARE_SIZE + 50
            y = row * self.SQUARE_SIZE + 50
            
            # Chọn ký tự
            symbol = self.PIECE_SYMBOLS[piece.piece_type]
            if piece.color == chess.BLACK:
                symbol = symbol.lower()
            
            # Chọn màu
            color = self.TEXT_COLOR if piece.color == chess.WHITE else (50, 50, 50)
            
            # Vẽ text
            text = self.font_large.render(symbol, True, color)
            text_rect = text.get_rect(center=(x + self.SQUARE_SIZE // 2,
                                             y + self.SQUARE_SIZE // 2))
            self.screen.blit(text, text_rect)
    
    def draw_info_panel(self):
        """Vẽ panel thông tin"""
        panel_x = 900
        
        # Game status
        if self.game_over:
            if self.game_result == 'checkmate':
                winner = "Đen" if self.board.turn == chess.WHITE else "Trắng"
                status = f"{winner} thắng!"
            elif self.game_result == 'stalemate':
                status = "Hòa (Stalemate)"
            else:
                status = "Game kết thúc"
            
            color = (255, 100, 100)
        else:
            status = "Trắng" if self.board.turn == chess.WHITE else "Đen"
            color = self.TEXT_COLOR
        
        status_text = self.font_medium.render(status, True, color)
        self.screen.blit(status_text, (panel_x, 50))
        
        # Move history
        y = 150
        history_text = self.font_small.render("Move History:", True, self.TEXT_COLOR)
        self.screen.blit(history_text, (panel_x, y))
        
        y += 40
        for i, move in enumerate(self.move_history[-10:]):  # Show last 10
            move_text = self.font_small.render(f"{i+1}. {move}", True, self.TEXT_COLOR)
            self.screen.blit(move_text, (panel_x, y))
            y += 30
        
        # Controls
        y = 600
        controls = [
            "Controls:",
            "Click: Select piece",
            "Right-click: Undo",
            "R: Reset",
            "Q: Quit"
        ]
        
        for text in controls:
            control_text = self.font_small.render(text, True, self.TEXT_COLOR)
            self.screen.blit(control_text, (panel_x, y))
            y += 30
    
    def draw(self):
        """Vẽ tất cả"""
        self.screen.fill(self.BG_COLOR)
        self.draw_board()
        self.draw_pieces()
        self.draw_info_panel()
        pygame.display.flip()
    
    def handle_click(self, pos: Tuple[int, int], button: int = 1):
        """
        Xử lý click chuột
        button: 1 (trái), 3 (phải)
        """
        if self.game_over:
            return
        
        # Check nếu click vào bàn cờ
        if pos[0] < 50 or pos[0] > 850 or pos[1] < 50 or pos[1] > 850:
            return
        
        col = (pos[0] - 50) // self.SQUARE_SIZE
        row = (pos[1] - 50) // self.SQUARE_SIZE
        square = (7 - row) * 8 + col
        
        if button == 3:  # Phải click = undo
            if self.move_history:
                self.board.pop()
                self.move_history.pop()
                self.selected_square = None
                self.legal_moves = []
            return
        
        # Nếu chưa chọn quân
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece is not None and piece.color == self.board.turn:
                self.selected_square = square
                # Lấy nước đi hợp lệ từ ô này
                self.legal_moves = [m for m in self.board.legal_moves 
                                  if m.from_square == square]
        else:
            # Đã chọn, bây giờ chọn ô đích
            move = chess.Move(self.selected_square, square)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move.uci())
                self.check_game_over()
                self.selected_square = None
                self.legal_moves = []
            else:
                self.selected_square = None
                self.legal_moves = []
    
    def check_game_over(self):
        """Check xem game kết thúc chưa"""
        if self.board.is_game_over():
            self.game_over = True
            
            if self.board.is_checkmate():
                self.game_result = 'checkmate'
            elif self.board.is_stalemate():
                self.game_result = 'stalemate'
            else:
                self.game_result = 'other'
    
    def ai_move(self):
        """AI đi"""
        if self.game_over or self.board.turn != self.ai_color:
            return
        
        if self.ai_mode == 'random':
            move = RandomEngine.get_best_move(self.board)
        elif self.ai_mode == 'minimax' and self.engine:
            move = self.engine.get_best_move(self.board)
        else:
            move = RandomEngine.get_best_move(self.board)
        
        if move:
            self.board.push(move)
            self.move_history.append(move.uci())
            self.check_game_over()
    
    def reset(self):
        """Reset game"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.game_result = None
        self.move_history = []
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos, event.button)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_q:
                        running = False
            
            # AI move
            self.ai_move()
            
            # Draw
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


def create_gui_with_engine(model_path: str = None, ai_color: chess.Color = chess.BLACK,
                          max_depth: int = 3) -> ChessGUI:
    """
    Tạo GUI với engine
    
    Args:
        model_path: Đường dẫn tới model weights
        ai_color: Màu AI
        max_depth: Độ sâu Minimax
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load network
    network = ValueNetwork(hidden_size=128)
    if model_path and Path(model_path).exists():
        network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã load model: {model_path}")
    else:
        print("Sử dụng untrained network")
    
    network.to(device)
    
    # Tạo engine
    engine = MinimaxEngine(network, device=device, max_depth=max_depth)
    
    # Tạo GUI
    gui = ChessGUI(engine=engine, ai_color=ai_color, ai_mode='minimax')
    
    return gui


if __name__ == "__main__":
    # Chơi với AI
    # Nếu model đã trained, load nó
    model_path = './models/chess_value_network.pth'
    
    gui = create_gui_with_engine(model_path=model_path, 
                                ai_color=chess.BLACK, 
                                max_depth=3)
    gui.run()
