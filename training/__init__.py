"""
Training Module
Contains trainer and training pipeline
"""

from .train import ChessTrainer, full_training_pipeline

__all__ = [
    'ChessTrainer',
    'full_training_pipeline'
]
