try:
    from chess_ai.board import board_state as board_state
    __all__ = ["board_state"]
except Exception:
    __all__ = []
from .board_state import *

__all__ = ['board_state']
