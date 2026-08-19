"""Модель ценообразования - пакет для прогнозирования цен."""

from price_prediction.data_loader import DataLoader
from price_prediction.database_manager import DatabaseManager
from price_prediction.feature_engineer import FeatureEngineer
from price_prediction.model_updater import ModelUpdater
from price_prediction.price_predictor import PricePredictor

__all__ = [
    "PricePredictor",
    "DataLoader",
    "DatabaseManager",
    "FeatureEngineer",
    "ModelUpdater",
]

__version__ = "1.0.0"
