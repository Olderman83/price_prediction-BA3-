"""Тесты для утилит."""

import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd

from price_prediction.utils import (calculate_metrics, load_model_metadata,
                                    safe_divide, save_model_metadata,
                                    setup_logging, validate_dataframe)


class TestUtils:
    """Тесты для утилит."""

    def test_setup_logging(self):
        """Тест настройки логирования."""
        # Проверка без файла
        setup_logging(log_level="DEBUG")
        logger = logging.getLogger("test")
        assert logger.level == logging.DEBUG

        # Проверка с файлом
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_file = f.name

        try:
            setup_logging(log_file=log_file, log_level="WARNING")
            assert os.path.exists(log_file)
        finally:
            os.unlink(log_file)

    def test_save_model_metadata(self):
        """Тест сохранения метаданных модели."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pkl")
            metadata = {"accuracy": 0.95, "loss": 0.1, "model_type": "ridge"}

            save_model_metadata(model_path, metadata)

            # Проверяем, что файл создан
            files = os.listdir(temp_dir)
            metadata_files = [f for f in files if f.startswith("model_metadata_")]
            assert len(metadata_files) > 0

            # Проверяем содержимое
            metadata_path = os.path.join(temp_dir, metadata_files[0])
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["accuracy"] == 0.95
            assert loaded["loss"] == 0.1
            assert loaded["model_type"] == "ridge"
            assert "timestamp" in loaded
            assert "created_at" in loaded

    def test_save_model_metadata_with_timestamp(self):
        """Тест сохранения с указанием временной метки."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pkl")
            metadata = {"accuracy": 0.95}
            timestamp = "20250101_120000"

            save_model_metadata(model_path, metadata, timestamp)

            files = os.listdir(temp_dir)
            expected_file = f"model_metadata_{timestamp}.json"
            assert expected_file in files

    def test_load_model_metadata(self):
        """Тест загрузки метаданных модели."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = os.path.join(temp_dir, "metadata.json")
            metadata = {"accuracy": 0.95, "loss": 0.1}

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

            loaded = load_model_metadata(metadata_path)
            assert loaded == metadata

    def test_validate_dataframe(self):
        """Тест проверки DataFrame."""
        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]}
        )

        # Все колонки присутствуют
        result = validate_dataframe(df, ["col1", "col2"])
        assert result is True

        # Некоторые колонки отсутствуют
        result = validate_dataframe(df, ["col1", "col2", "col4"])
        assert result is False

        # Пустой список
        result = validate_dataframe(df, [])
        assert result is True

    def test_calculate_metrics(self):
        """Тест расчета метрик."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = calculate_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics

        # Проверка значений
        assert metrics["mae"] > 0
        assert metrics["r2"] > 0

    def test_calculate_metrics_with_zeros(self):
        """Тест расчета метрик с нулевыми значениями."""
        y_true = np.array([1, 0, 3, 0, 5])
        y_pred = np.array([1.1, 0.1, 2.9, 0.2, 4.8])

        metrics = calculate_metrics(y_true, y_pred)
        assert "mape" in metrics
        # MAPE должен быть конечным
        assert metrics["mape"] != float("inf")

    def test_calculate_metrics_perfect_prediction(self):
        """Тест расчета метрик с идеальным предсказанием."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == 0
        assert metrics["mse"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0
        assert metrics["mape"] == 0

    def test_safe_divide(self):
        """Тест безопасного деления."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, default=999) == 999
        assert safe_divide(0, 5) == 0

    def test_calculate_metrics_with_nan(self):
        """Тест расчета метрик с NaN."""
        y_true = np.array([1, 2, np.nan, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.0, 3.8, 5.1])

        # Функция должна работать, но значения с NaN будут влиять
        metrics = calculate_metrics(y_true, y_pred)
        assert "mae" in metrics
        # Проверка, что результат не NaN
        assert not np.isnan(metrics["mae"])
