"""Вспомогательные функции и утилиты."""

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    """
    Настройка логирования.

    Args:
        log_file: Путь к файлу лога (опционально)
        log_level: Уровень логирования
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )


def save_model_metadata(
    model_path: str, metadata: dict, timestamp: Optional[str] = None
) -> None:
    """
    Сохранение метаданных модели.

    Args:
        model_path: Путь к модели
        metadata: Метаданные модели
        timestamp: Временная метка
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata_file = os.path.join(
        os.path.dirname(model_path), f"model_metadata_{timestamp}.json"
    )

    metadata["timestamp"] = timestamp
    metadata["created_at"] = datetime.now().isoformat()

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_model_metadata(metadata_path: str) -> dict:
    """
    Загрузка метаданных модели.

    Args:
        metadata_path: Путь к файлу метаданных

    Returns:
        dict: Метаданные модели
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataframe(df: "pd.DataFrame", required_columns: list) -> bool:
    """
    Проверка наличия необходимых колонок в DataFrame.

    Args:
        df: DataFrame для проверки
        required_columns: Список необходимых колонок

    Returns:
        bool: True если все колонки присутствуют
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Отсутствуют колонки: {missing_columns}")
        return False
    return True


def calculate_metrics(y_true: "np.ndarray", y_pred: "np.ndarray") -> dict:
    """
    Расчет метрик качества модели.

    Args:
        y_true: Фактические значения
        y_pred: Предсказанные значения

    Returns:
        dict: Словарь с метриками
    """
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

    # Вычисляем MAPE (Mean Absolute Percentage Error)
    y_true_nonzero = y_true[y_true != 0]
    y_pred_nonzero = y_pred[y_true != 0]
    if len(y_true_nonzero) > 0:
        mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        metrics["mape"] = float(mape)
    else:
        metrics["mape"] = float("inf")

    return metrics


def safe_divide(a, b, default=0):
    """
    Безопасное деление с обработкой деления на ноль.

    Args:
        a: Числитель
        b: Знаменатель
        default: Значение по умолчанию при делении на ноль

    Returns:
        float: Результат деления
    """
    if b == 0:
        return default
    return a / b
