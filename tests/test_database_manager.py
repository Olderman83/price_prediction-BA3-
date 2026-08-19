"""Тесты для DatabaseManager."""

import json
import os
import tempfile

import pandas as pd
import pytest

from price_prediction.database_manager import DatabaseManager


class TestDatabaseManager:
    """Тесты для DatabaseManager."""

    @pytest.fixture
    def temp_db_path(self):
        """Создание временной БД."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

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
    def db_manager(self, temp_db_path):
        """Создание менеджера БД."""
        return DatabaseManager(temp_db_path)

    def test_init(self, temp_db_path):
        """Тест инициализации."""
        manager = DatabaseManager(temp_db_path)
        assert manager.db_path == temp_db_path

        # Проверка создания таблиц
        with manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "price_history" in tables
            assert "model_metadata" in tables

    def test_insert_batch(self, db_manager, sample_data):
        """Тест вставки данных пачкой."""
        count = db_manager.insert_batch(sample_data)
        assert count == 5

        # Проверка что данные вставлены
        df = db_manager.get_all_data()
        assert len(df) == 5
        assert set(df.columns) == {
            "price",
            "count",
            "add_cost",
            "company",
            "product",
            "created_at",
        }

    def test_insert_batch_empty(self, db_manager):
        """Тест вставки пустого DataFrame."""
        df_empty = pd.DataFrame()
        count = db_manager.insert_batch(df_empty)
        assert count == 0

    def test_insert_batch_missing_columns(self, db_manager):
        """Тест вставки с отсутствующими колонками."""
        data = pd.DataFrame({"price": [100, 150], "count": [10, 20]})

        with pytest.raises(ValueError):
            db_manager.insert_batch(data)

    def test_get_all_data(self, db_manager, sample_data):
        """Тест получения всех данных."""
        db_manager.insert_batch(sample_data)

        df = db_manager.get_all_data()
        assert len(df) == 5
        assert "created_at" in df.columns

    def test_get_all_data_with_filters(self, db_manager, sample_data):
        """Тест получения данных с фильтрами."""
        db_manager.insert_batch(sample_data)

        df_company_a = db_manager.get_all_data(company="A")
        assert len(df_company_a) == 2  # Две записи с компанией A

        df_product_x = db_manager.get_all_data(product="X")
        assert len(df_product_x) == 2  # Две записи с продуктом X

    def test_get_all_data_with_limit(self, db_manager, sample_data):
        """Тест получения данных с лимитом."""
        db_manager.insert_batch(sample_data)

        df_limited = db_manager.get_all_data(limit=2)
        assert len(df_limited) == 2

    def test_get_latest_data(self, db_manager, sample_data):
        """Тест получения последних данных."""
        db_manager.insert_batch(sample_data)

        df_latest = db_manager.get_latest_data(limit=3)
        assert len(df_latest) == 3

    def test_get_statistics(self, db_manager, sample_data):
        """Тест получения статистики."""
        db_manager.insert_batch(sample_data)

        stats = db_manager.get_statistics()

        assert "total_records" in stats
        assert stats["total_records"] == 5
        assert "company_stats" in stats
        assert "product_stats" in stats

        # Проверка статистики по компаниям
        company_a_stats = [s for s in stats["company_stats"] if s["company"] == "A"]
        assert len(company_a_stats) == 1
        assert company_a_stats[0]["count"] == 2
        assert company_a_stats[0]["avg_price"] == 150.0

    def test_clear_data(self, db_manager, sample_data):
        """Тест очистки данных."""
        db_manager.insert_batch(sample_data)
        assert len(db_manager.get_all_data()) == 5

        db_manager.clear_data()
        assert len(db_manager.get_all_data()) == 0

    def test_save_model_metadata(self, db_manager):
        """Тест сохранения метаданных модели."""
        features = ["count", "add_cost", "company_encoded"]
        metrics = {"r2": 0.85, "mae": 10.5}

        id_ = db_manager.save_model_metadata(
            model_version="1.0.0",
            training_count=100,
            features=features,
            metrics=metrics,
        )

        assert id_ is not None

        # Проверка сохранения
        with db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_metadata WHERE id = ?", (id_,))
            row = cursor.fetchone()

            assert row["model_version"] == "1.0.0"
            assert row["training_data_count"] == 100
            assert json.loads(row["features"]) == features
            assert json.loads(row["metrics"]) == metrics

    def test_context_manager_rollback(self, temp_db_path):
        """Тест отката транзакции."""
        manager = DatabaseManager(temp_db_path)

        # Вставляем данные
        data = pd.DataFrame(
            {
                "price": [100],
                "count": [10],
                "add_cost": [500],
                "company": ["A"],
                "product": ["X"],
            }
        )
        manager.insert_batch(data)

        # Симулируем ошибку внутри транзакции
        try:
            with manager._get_connection() as conn:
                conn.execute("INSERT INTO price_history (price) VALUES (?)", (999,))
                raise Exception("Test error")
        except Exception:
            pass

        # Проверяем, что данные не были вставлены
        df = manager.get_all_data()
        assert len(df) == 1  # Только первая вставка

    def test_connection_error_handling(self, temp_db_path):
        """Тест обработки ошибок подключения."""
        # Создаем менеджер с некорректным путем
        manager = DatabaseManager("/nonexistent/dir/test.db")

        # Пытаемся выполнить операцию
        data = pd.DataFrame(
            {
                "price": [100],
                "count": [10],
                "add_cost": [500],
                "company": ["A"],
                "product": ["X"],
            }
        )

        # Должна быть ошибка, но не критическая
        with pytest.raises(Exception):
            manager.insert_batch(data)
