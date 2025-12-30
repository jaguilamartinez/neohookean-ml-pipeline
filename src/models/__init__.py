"""
Neural network models for Neo-Hookean ML pipeline.

This package contains:
- base_model: Baseline black-box feedforward network
- pann_model: Physics-informed neural network (PANN)
"""

from .base_model import (
    build_model as build_base_model,
    train_model,
    evaluate_model,
    save_results,
    load_trained_model,
    run_training as run_base_training
)

# PANN model
from .pann_model import (
    build_pann_model,
    train_pann_model,
    run_pann_training,
    PANNModel,
    ICNNEnergy
)

__all__ = [
    'build_base_model',
    'train_model',
    'evaluate_model',
    'save_results',
    'load_trained_model',
    'run_base_training',
    'build_pann_model',
    'train_pann_model',
    'run_pann_training',
    'PANNModel',
    'ICNNEnergy',
]
