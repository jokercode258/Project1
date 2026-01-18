try:
    from chess_ai.network.value_network import value_network as value_network
    __all__ = ["value_network"]
except Exception:
    __all__ = []
from .value_network import *

__all__ = ['value_network']
