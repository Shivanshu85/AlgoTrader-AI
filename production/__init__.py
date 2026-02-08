"""
Stock Predictor Production
==========================

Production-grade stock price prediction platform with LSTM and attention mechanisms.

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "ML Engineering Team"
__email__ = "team@example.com"
__license__ = "MIT"

# Make key modules available at package level
from production.utils.logging import setup_logging

# Setup logging when package is imported
setup_logging()

__all__ = [
    "setup_logging",
]
