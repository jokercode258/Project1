import pygame
import chess
import numpy as np
from typing import Optional, Tuple
from chess_ai.board.board_state import BoardState
from chess_ai.engine.minimax_engine import MinimaxEngine, RandomEngine
from chess_ai.network.value_network import ValueNetwork, ImprovedValueNetwork
import torch
from pathlib import Path
import os


class ChessGUI:
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
    
    # Piece to image filename mapping
    PIECE_IMAGES = {
        (chess.PAWN, chess.WHITE): 'wp.png',
        (chess.KNIGHT, chess.WHITE): 'wn.png',
        (chess.BISHOP, chess.WHITE): 'wb.png',
        (chess.ROOK, chess.WHITE): 'wr.png',
        (chess.QUEEN, chess.WHITE): 'wq.png',
        (chess.KING, chess.WHITE): 'wk.png',
        (chess.PAWN, chess.BLACK): 'bp.png',
        (chess.KNIGHT, chess.BLACK): 'bn.png',
        (chess.BISHOP, chess.BLACK): 'bb.png',
        (chess.ROOK, chess.BLACK): 'br.png',
        (chess.QUEEN, chess.BLACK): 'bq.png',
        (chess.KING, chess.BLACK): 'bk.png',
    }
    
    def __init__(self, engine: MinimaxEngine = None, ai_color: chess.Color = chess.BLACK,
                 ai_mode: str = 'minimax'):
        self.waiting_for_ai = False
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
        # Whether we've already printed the final move list for this game
        self.printed_game_over = False
        
        # Track last AI move evaluation
        self.last_ai_move = None
        self.last_ai_score = None
        self.last_ai_nodes = None
        
        # Load piece images
        self.piece_images = self._load_piece_images()

    def _load_piece_images(self) -> dict:
        piece_images = {}
        # Thư mục chứa file gui.py (gui_module/)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # assets/ ở cùng level với gui_module/
        assets_dir = os.path.join(os.path.dirname(base_dir), 'assets')

        print(" Assets dir:", assets_dir)

        for (piece_type, color), filename in self.PIECE_IMAGES.items():
            filepath = os.path.join(assets_dir, filename)

            try:
                img = pygame.image.load(filepath).convert_alpha()
                size = int(self.SQUARE_SIZE * 0.9)
                img = pygame.transform.smoothscale(img, (size, size))
                piece_images[(piece_type, color)] = img
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        return piece_images

    
    def _get_piece_symbol(self, piece: chess.Piece) -> str:
        """
        Get Unicode symbol for fallback (if image not loaded)
        """
        symbols = {
            chess.PAWN: '♟' if piece.color == chess.BLACK else '♙',
            chess.KNIGHT: '♞' if piece.color == chess.BLACK else '♘',
            chess.BISHOP: '♝' if piece.color == chess.BLACK else '♗',
            chess.ROOK: '♜' if piece.color == chess.BLACK else '♖',
            chess.QUEEN: '♛' if piece.color == chess.BLACK else '♕',
            chess.KING: '♚' if piece.color == chess.BLACK else '♔',
        }
        return symbols.get(piece.piece_type, '?')
    
    def draw_board(self):
        """Vẽ bàn cờ 8x8"""
        for row in range(8):
            for col in range(8):
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE
                
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
            x = col * self.SQUARE_SIZE
            y = row * self.SQUARE_SIZE
            pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR,
                           (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 3)
    
    def draw_pieces(self):
        """Vẽ các quân cờ từ hình ảnh PNG"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            
            row = 7 - (square // 8)
            col = square % 8
            x = col * self.SQUARE_SIZE
            y = row * self.SQUARE_SIZE
            
            # Get piece image
            piece_key = (piece.piece_type, piece.color)
            if piece_key in self.piece_images:
                # Draw image
                img = self.piece_images[piece_key]
                # Center image in square
                img_rect = img.get_rect(center=(x + self.SQUARE_SIZE // 2,
                                               y + self.SQUARE_SIZE // 2))
                self.screen.blit(img, img_rect)
            else:
                # Fallback: draw text if image not available
                symbol = self._get_piece_symbol(piece)
                text = self.font_large.render(symbol, True, self.TEXT_COLOR)
                text_rect = text.get_rect(center=(x + self.SQUARE_SIZE // 2,
                                                 y + self.SQUARE_SIZE // 2))
                self.screen.blit(text, text_rect)
    
    def draw_info_panel(self):
        """Vẽ panel thông tin"""
        panel_x = 900
        
        # Game status
        if self.game_over:
            if self.game_result == 'checkmate':
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                status = f"{winner} Win!"
            elif self.game_result == 'stalemate':
                status = "Draw (Stalemate)"
            else:
                status = "Game over"
            
            color = (255, 100, 100)
        else:
            status = "White" if self.board.turn == chess.WHITE else "Black"
            color = self.TEXT_COLOR
        
        status_text = self.font_medium.render(status, True, color)
        self.screen.blit(status_text, (panel_x, 50))
        
        # Last AI move evaluation
        y = 100
        if self.last_ai_move is not None:
            ai_move_text = self.font_small.render(f"Last AI Move: {self.last_ai_move}", True, (100, 200, 255))
            self.screen.blit(ai_move_text, (panel_x, y))
            y += 30
            
            if self.last_ai_score is not None:
                score_text = self.font_small.render(f"Score: {self.last_ai_score:.4f}", True, (150, 200, 100))
                self.screen.blit(score_text, (panel_x, y))
                y += 30
            
            if self.last_ai_nodes is not None:
                nodes_text = self.font_small.render(f"Nodes: {self.last_ai_nodes}", True, (200, 150, 100))
                self.screen.blit(nodes_text, (panel_x, y))
                y += 30
        
        # Move history
        y = y + 20 if self.last_ai_move is not None else 150
        history_text = self.font_small.render("Move History:", True, self.TEXT_COLOR)
        self.screen.blit(history_text, (panel_x, y))
        
        y += 40
        # Show up to the first 12 moves in the GUI (then stop adding more)
        for i, move in enumerate(self.move_history[:12], start=1):
            move_text = self.font_small.render(f"{i}. {move}", True, self.TEXT_COLOR)
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
        pygame.draw.rect(self.screen, (255,0,0), (0,0,100,100))

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
        
        col = (pos[0]) // self.SQUARE_SIZE
        row = (pos[1]) // self.SQUARE_SIZE
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

            piece = self.board.piece_at(self.selected_square)
            # Xử lý thăng quân tự động
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(square)
                if to_rank == 0 or to_rank == 7:
                    move = chess.Move(
                        self.selected_square,
                        square,
                        promotion=chess.QUEEN
                    )

            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move.uci())
                self.check_game_over()
                self.selected_square = None
                self.legal_moves = []
                self.waiting_for_ai = True
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
            # Print full move list to terminal once when game ends
            if not getattr(self, 'printed_game_over', False):
                self.print_full_moves()
                self.printed_game_over = True
    
    def ai_move(self):
        """AI đi"""
        if self.game_over or self.board.turn != self.ai_color:
            return
        
        move = None
        score = None
        nodes = None
        
        if self.ai_mode == 'random':
            move = RandomEngine.get_best_move(self.board)
        elif self.ai_mode == 'minimax' and self.engine:
            # Get best move with score (iterative deepening with TT, move ordering, quiescence, etc.)
            move, score = self.engine.get_best_move_with_score(self.board)
            nodes = self.engine.nodes_evaluated
        else:
            move = RandomEngine.get_best_move(self.board)
        
        if move:
            # Print evaluation to console
            print(f"\n AI Move: {move}")
            if score is not None:
                print(f"   Score: {score:.4f}")
            if nodes is not None:
                print(f"   Nodes evaluated: {nodes}")
            
            # Store for display
            self.last_ai_move = move.uci()
            self.last_ai_score = score
            self.last_ai_nodes = nodes
            
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
        self.last_ai_move = None
        self.last_ai_score = None
        self.last_ai_nodes = None
        self.printed_game_over = False

    def print_full_moves(self, up_to: Optional[int] = None) -> None:
        """Print all moves (both white and black) to the terminal.

        If `up_to` is provided, print moves only up to that many entries.
        Moves are printed in paired form: '1. e2e4 e7e5'.
        """
        moves = self.move_history if up_to is None else self.move_history[:up_to]
        if not moves:
            print("No moves to show.")
            return

        print("\nFull move list:")
        for i in range(0, len(moves), 2):
            move_no = i // 2 + 1
            white = moves[i]
            black = moves[i+1] if i+1 < len(moves) else ""
            if black:
                print(f"{move_no}. {white} {black}")
            else:
                print(f"{move_no}. {white}")
    
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
                        # Print all moves up to this point, then quit
                        self.print_full_moves()
                        running = False
            
            # AI move
            if self.waiting_for_ai and self.board.turn == self.ai_color:
                pygame.time.delay(300)  # cho người chơi thấy nước đi
                self.ai_move()
                self.waiting_for_ai = False
            
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
    
    # Auto-detect network type from checkpoint
    network = None
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)

        # Detect network type from checkpoint structure
        network_type = None
        state_dict = None
        if isinstance(checkpoint, dict):
            # common checkpoint formats: {'model_state_dict': {...}, 'optimizer_state_dict': ...}
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                network_type = checkpoint.get('network_type', None)
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Infer network type from state_dict keys if not provided
        if network_type is None:
            if isinstance(state_dict, dict) and any(k.startswith('conv') for k in state_dict.keys()):
                network_type = 'ImprovedValueNetwork'
            else:
                network_type = 'ValueNetwork'
        
        # Instantiate correct network
        if network_type == 'ImprovedValueNetwork':
            network = ImprovedValueNetwork(dropout=0.3)
            print(f"Loaded ImprovedValueNetwork from {model_path}")
        else:
            network = ValueNetwork(hidden_size=128, dropout=0.3)
            print(f"Loaded ValueNetwork from {model_path}")
        
        # Load state dict
        network.load_state_dict(state_dict)
        print(f"Đã load model: {model_path}")
    else:
        # Default network if no model provided
        network = ImprovedValueNetwork(dropout=0.3)
        print("Sử dụng untrained ImprovedValueNetwork")
    
    network.to(device)
    
    # Tạo engine
    engine = MinimaxEngine(network, device=device, max_depth=max_depth)
    
    # Tạo GUI
    gui = ChessGUI(engine=engine, ai_color=ai_color, ai_mode='minimax')
    
    return gui


if __name__ == "__main__":
    # Chơi với AI
    # Nếu model đã trained, load nó
    model_path = './data/chess_value_network.pth'
    
    gui = create_gui_with_engine(model_path=model_path, 
                                ai_color=chess.BLACK, 
                                max_depth=3)
    gui.run()
