import numpy as np
import chess
from typing import Tuple, Optional
from ..board.board_state import BoardState
from ..network.value_network import ValueNetwork
import torch
import time
from collections import defaultdict


class MinimaxEngine:
    """
    Minimax với Alpha-Beta Pruning
    Sử dụng Neural Network để đánh giá node lá
    """
    
    def __init__(self, network: ValueNetwork, device: torch.device = None, 
                 max_depth: int = 4):
        self.network = network
        self.device = device if device else torch.device('cpu')
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.transposition_table = {}
        self.history_scores = defaultdict(int)
        self.deadline = None

    def _is_forcing_position(self, board: chess.Board) -> bool:
        return board.is_check()

    def _evaluate_material(self, board: chess.Board) -> float:
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        white_material = 0.0
        black_material = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
        max_material = 30.0
        material_diff = white_material - black_material
        normalized = material_diff / max_material
        return max(-1.0, min(1.0, normalized))

    def evaluate_position(self, board: chess.Board, allow_in_forcing: bool = False) -> float:
        if board.is_game_over():
            result = BoardState.get_game_result(board)
            if result is not None:
                return result
        material_eval = self._evaluate_material(board)
        try:
            tensor = BoardState.board_to_tensor(board)
            network_eval = self.network.evaluate_position(tensor, self.device)
        except Exception:
            network_eval = material_eval
        alpha = 0.7
        beta = 0.3
        value = alpha * network_eval + beta * material_eval
        return max(-1.0, min(1.0, value))

    def minimax(self, board: chess.Board, depth: int, maximizing: bool,
                alpha: float = -float('inf'), beta: float = float('inf'), seen_fens: set = None) -> Tuple[float, Optional[chess.Move]]:
        if seen_fens is None:
            seen_fens = {board.fen()}
        fen = board.fen()
        tt_entry = self.transposition_table.get(fen)
        if tt_entry is not None and tt_entry.get('depth', -1) >= depth:
            return tt_entry['value'], tt_entry.get('best_move')
        if depth == 0 or board.is_game_over():
            result = BoardState.get_game_result(board)
            if result is not None:
                return result, None
            if depth == 0 and self._is_forcing_position(board):
                value = self.quiescence(board, alpha, beta, maximizing, seen_fens)
            else:
                value = self.evaluate_position(board)
            self.nodes_evaluated += 1
            self.transposition_table[fen] = {'value': value, 'best_move': None, 'depth': depth}
            return value, None
        best_move = None
        if maximizing:
            max_eval = -float('inf')
            piece_values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
            moves = list(board.legal_moves)
            scored_moves = []
            for mv in moves:
                board.push(mv)
                score = 0.0
                if board.is_checkmate():
                    score += 1e9
                    board.pop()
                    scored_moves.append((score, mv))
                    continue
                if board.is_check():
                    score += 1000.0
                if board.is_capture(mv):
                    victim = board.piece_at(mv.to_square)
                    attacker = board.piece_at(mv.from_square)
                    if victim and attacker:
                        victim_val = piece_values.get(victim.piece_type, 0)
                        attacker_val = piece_values.get(attacker.piece_type, 0)
                        score += 100.0 * victim_val - 10.0 * attacker_val
                if mv.promotion is not None:
                    promote_val = piece_values.get(mv.promotion, 0)
                    score += 80.0 + promote_val * 10.0
                try:
                    quick_eval = self.evaluate_material(board)
                    score += 10.0 * quick_eval
                except Exception:
                    score += 0.0
                board.pop()
                score += self.history_scores.get(mv.uci(), 0) * 0.01
                scored_moves.append((score, mv))
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            for _, move in scored_moves:
                if self.deadline is not None and time.time() > self.deadline:
                    break
                board.push(move)
                next_fen = board.fen()
                if board.is_checkmate():
                    max_eval = float('inf')
                    best_move = move
                    board.pop()
                    self.transposition_table[fen] = {'value': max_eval, 'best_move': best_move, 'depth': depth}
                    return max_eval, best_move
                if next_fen in seen_fens:
                    board.pop()
                    continue
                new_seen = set(seen_fens)
                new_seen.add(next_fen)
                eval_score, _ = self.minimax(board, depth - 1, False, alpha, beta, new_seen)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.history_scores[move.uci()] += 1 << depth
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            piece_values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
            moves = list(board.legal_moves)
            scored_moves = []
            for mv in moves:
                board.push(mv)
                score = 0.0
                if board.is_checkmate():
                    score -= 1e9
                    board.pop()
                    scored_moves.append((score, mv))
                    continue
                if board.is_check():
                    score += 1000.0
                if board.is_capture(mv):
                    victim = board.piece_at(mv.to_square)
                    attacker = board.piece_at(mv.from_square)
                    if victim and attacker:
                        victim_val = piece_values.get(victim.piece_type, 0)
                        attacker_val = piece_values.get(attacker.piece_type, 0)
                        score += 100.0 * victim_val - 10.0 * attacker_val
                if mv.promotion is not None:
                    promote_val = piece_values.get(mv.promotion, 0)
                    score += 80.0 + promote_val * 10.0
                try:
                    quick_eval = self.evaluate_position(board)
                    score += 10.0 * quick_eval
                except Exception:
                    score += 0.0
                board.pop()
                score += self.history_scores.get(mv.uci(), 0) * 0.01
                scored_moves.append((score, mv))
            scored_moves.sort(key=lambda x: x[0])
            for _, move in scored_moves:
                if self.deadline is not None and time.time() > self.deadline:
                    break
                board.push(move)
                next_fen = board.fen()
                if board.is_checkmate():
                    min_eval = -float('inf')
                    best_move = move
                    board.pop()
                    self.transposition_table[fen] = {'value': min_eval, 'best_move': best_move, 'depth': depth}
                    return min_eval, best_move
                if next_fen in seen_fens:
                    board.pop()
                    continue
                new_seen = set(seen_fens)
                new_seen.add(next_fen)
                eval_score, _ = self.minimax(board, depth - 1, True, alpha, beta, new_seen)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.history_scores[move.uci()] += 1 << depth
                    break
            self.transposition_table[fen] = {'value': min_eval, 'best_move': best_move, 'depth': depth}
            return min_eval, best_move

    def quiescence(self, board: chess.Board, alpha: float, beta: float, maximizing: bool, seen_fens: set, depth: int = 0) -> float:
        if depth > 10:
            return self.evaluate_position(board, allow_in_forcing=True)
        stand_pat = self.evaluate_position(board, allow_in_forcing=True)
        self.nodes_evaluated += 1
        if maximizing:
            if stand_pat >= beta:
                return stand_pat
            if alpha < stand_pat:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return stand_pat
            if beta > stand_pat:
                beta = stand_pat
        forcing_moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        scored = []
        for mv in forcing_moves:
            board.push(mv)
            nf = board.fen()
            if nf in seen_fens:
                board.pop()
                continue
            try:
                q = self.evaluate_position(board, allow_in_forcing=True)
            except Exception:
                q = 0.0
            board.pop()
            scored.append((q, mv))
        if maximizing:
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, mv in scored:
                board.push(mv)
                nf = board.fen()
                if nf in seen_fens:
                    board.pop()
                    continue
                new_seen = set(seen_fens)
                new_seen.add(nf)
                score = self.quiescence(board, alpha, beta, False, new_seen, depth + 1)
                board.pop()
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return alpha
            return alpha
        else:
            scored.sort(key=lambda x: x[0])
            for _, mv in scored:
                board.push(mv)
                nf = board.fen()
                if nf in seen_fens:
                    board.pop()
                    continue
                new_seen = set(seen_fens)
                new_seen.add(nf)
                score = self.quiescence(board, alpha, beta, True, new_seen, depth + 1)
                board.pop()
                if score < beta:
                    beta = score
                if beta <= alpha:
                    return beta
            return beta

    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
        self.nodes_evaluated = 0
        best_move, _ = self.get_best_move_with_score(board)
        return best_move

    def get_best_move_with_score(self, board: chess.Board, time_limit: float = None) -> Tuple[Optional[chess.Move], float]:
        if board.is_game_over():
            return None, None
        self.nodes_evaluated = 0
        is_maximizing = board.turn == chess.WHITE
        self.deadline = time.time() + time_limit if time_limit is not None else None
        best_move = None
        best_score = None
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self.minimax(board, depth, is_maximizing)
            except Exception:
                break
            if move is not None:
                best_move = move
                best_score = score
            if self.deadline is not None and time.time() > self.deadline:
                break
        self.deadline = None
        if best_move is None:
            legal = list(board.legal_moves)
            if legal:
                best_move = legal[0]
                best_score = 0.0
        return best_move, best_score

    def set_max_depth(self, depth: int):
        self.max_depth = depth


class RandomEngine:
    @staticmethod
    def get_best_move(board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        return np.random.choice(moves)


class HybridEngine:
    def __init__(self, network: ValueNetwork = None, max_depth: int = 4, device: torch.device = None):
        self.network = network
        self.minimax_engine = MinimaxEngine(network, device, max_depth) if network else None
        self.device = device

    def get_best_move(self, board: chess.Board, mode: str = 'minimax') -> Optional[chess.Move]:
        if mode == 'random':
            return RandomEngine.get_best_move(board)
        elif mode == 'minimax':
            if self.minimax_engine is None:
                raise ValueError("Network chưa được khởi tạo cho minimax")
            return self.minimax_engine.get_best_move(board)
        else:
            raise ValueError(f"Mode không hợp lệ: {mode}")
