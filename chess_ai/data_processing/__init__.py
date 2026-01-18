try:
    from chess_ai.data_processing import pgn_downloader as pgn_downloader
    from chess_ai.data_processing import pgn_processor as pgn_processor
    __all__ = ["pgn_downloader", "pgn_processor"]
except Exception:
    __all__ = []
from .pgn_downloader import *
from .pgn_processor import *

__all__ = ['pgn_downloader', 'pgn_processor']
