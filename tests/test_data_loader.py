"""Тесты для DataLoader."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from price_prediction.data_loader import DataLoader


class TestDataLoader:
    """Тесты для DataLoader."""

    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных."""
        return pd.DataFrame(
            {
                "price": [100, 150, 200, 250, 300],
                "count": [10, 20, 30, 40, 50],
                "add_cost": [500, 600, 700, 800, 900],
                "company": ["A", "B", "A", "B", "C"],
                "product": ["X", "Y", "X", "Y", "Z"],
            }
        )

    @pytest.fixture
    def sample_data_with_missing(self):
        """Создание тестовых данных с пропусками."""
        return pd.DataFrame(
            {
                "price": [100, 150, 200, np.nan, 300],
                "count": [10, 20, np.nan, 40, 50],
                "add_cost": [500, 600, 700, 800, np.nan],
                "company": ["A", "B", None, "B", "C"],
                "product": ["X", "Y", "X", None, "Z"],
            }
        )

    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Создание временного CSV файла."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_init(self):
        """Тест инициализации."""
        loader = DataLoader()
        assert loader.filepath is None
        assert loader.data is None
        assert loader._preprocessing_params == {}

        loader = DataLoader("test.csv")
        assert loader.filepath == "test.csv"

    def test_load_data(self, temp_csv_file):
        """Тест загрузки данных."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()

        assert df is not None
        assert len(df) == 5
        assert list(df.columns) == ["price", "count", "add_cost", "company", "product"]

    def test_load_data_no_file(self):
        """Тест загрузки с несуществующим файлом."""
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_load_data_no_path(self):
        """Тест загрузки без указания пути."""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.load_data()

    def test_preprocess_data(self, sample_data):
        """Тест предобработки данных."""
        loader = DataLoader()
        df_processed = loader.preprocess_data(sample_data, fit=True)

        assert len(df_processed) == 5
        assert "price" in df_processed.columns
        assert not df_processed.isnull().any().any()

    def test_preprocess_data_with_missing(self, sample_data_with_missing):
        """Тест предобработки данных с пропусками."""
        loader = DataLoader()
        df_processed = loader.preprocess_data(sample_data_with_missing, fit=True)

        assert len(df_processed) == 5
        assert not df_processed.isnull().any().any()

        # Проверка заполнения пропусков
        assert df_processed["price"].isnull().sum() == 0
        assert df_processed["count"].isnull().sum() == 0
        assert df_processed["add_cost"].isnull().sum() == 0

    def test_preprocess_data_duplicates(self):
        """Тест удаления дубликатов."""
        data = pd.DataFrame(
            {
                "price": [100, 100, 150],
                "count": [10, 10, 20],
                "add_cost": [500, 500, 600],
                "company": ["A", "A", "B"],
                "product": ["X", "X", "Y"],
            }
        )

        loader = DataLoader()
        df_processed = loader.preprocess_data(data, fit=True)

        assert len(df_processed) == 2  # Один дубликат удален

    def test_preprocess_data_missing_columns(self):
        """Тест с отсутствующими колонками."""
        data = pd.DataFrame(
            {
                "price": [100, 150],
                "count": [10, 20],
                # Нет add_cost, company, product
            }
        )

        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.preprocess_data(data, fit=True)

    def test_handle_outliers_iqr(self):
        """Тест обработки выбросов методом IQR."""
        data = pd.DataFrame(
            {
                "price": [100, 150, 200, 250, 1000],  # 1000 - выброс
                "count": [10, 20, 30, 40, 50],
                "add_cost": [500, 600, 700, 800, 900],
                "company": ["A", "B", "A", "B", "C"],
                "product": ["X", "Y", "X", "Y", "Z"],
            }
        )

        loader = DataLoader()
        df_processed = loader.preprocess_data(data, fit=True)

        # Проверяем, что выброс обработан (цена 1000 должна быть уменьшена)
        assert df_processed["price"].max() < 1000

    def test_get_preprocessing_params(self, sample_data):
        """Тест получения параметров предобработки."""
        loader = DataLoader()
        loader.preprocess_data(sample_data, fit=True)

        params = loader.get_preprocessing_params()
        assert "price_median" in params
        assert "count_median" in params
        assert "add_cost_median" in params
        assert "company_mode" in params
        assert "product_mode" in params

    def test_set_preprocessing_params(self):
        """Тест установки параметров предобработки."""
        loader = DataLoader()
        params = {"test_param": "test_value"}
        loader.set_preprocessing_params(params)

        assert loader.get_preprocessing_params() == params

    def test_get_summary_statistics(self, sample_data):
        """Тест получения сводной статистики."""
        loader = DataLoader()
        loader.data = sample_data

        stats = loader.get_summary_statistics()

        assert "total_rows" in stats
        assert stats["total_rows"] == 5
        assert "columns" in stats
        assert "numeric_stats" in stats
        assert "categorical_stats" in stats
        assert "price" in stats["numeric_stats"]
        assert "company" in stats["categorical_stats"]

    def test_get_summary_statistics_no_data(self):
        """Тест получения статистики без данных."""
        loader = DataLoader()
        stats = loader.get_summary_statistics()
        assert stats == {}

    def test_preprocess_data_fit_false(self, sample_data_with_missing):
        """Тест предобработки с fit=False."""
        loader = DataLoader()

        # Сначала обучаем
        loader.preprocess_data(sample_data_with_missing, fit=True)
        params = loader.get_preprocessing_params()

        # Затем применяем
        new_loader = DataLoader()
        new_loader.set_preprocessing_params(params)
        df_processed = new_loader.preprocess_data(sample_data_with_missing, fit=False)

        assert not df_processed.isnull().any().any()
        assert len(df_processed) == 5
