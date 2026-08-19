"""Тесты для ModelUpdater."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from price_prediction.model_updater import ModelUpdater
from price_prediction.price_predictor import PricePredictor


class TestModelUpdater:
    """Тесты для ModelUpdater."""

    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame(
            {
                "price": 100 + np.random.randn(n) * 50,
                "count": np.random.randint(10, 100, n),
                "add_cost": np.random.randint(100, 1000, n),
                "company": np.random.choice(["A", "B", "C"], n),
                "product": np.random.choice(["X", "Y", "Z"], n),
            }
        )
        return data

    @pytest.fixture
    def temp_dir(self):
        """Создание временной директории."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def predictor(self, temp_dir):
        """Создание предиктора."""
        db_path = os.path.join(temp_dir, "test.db")
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        return PricePredictor(db_path=db_path, model_dir=model_dir)

    @pytest.fixture
    def updater(self, predictor):
        """Создание обновлятора."""
        return ModelUpdater(predictor=predictor)

    def test_init(self, predictor):
        """Тест инициализации."""
        updater = ModelUpdater(predictor=predictor)
        assert updater.predictor == predictor
        assert updater.is_running is False
        assert updater.last_update is None

    def test_update_model(self, updater, sample_data, temp_dir):
        """Тест обновления модели."""
        # Сохраняем данные в CSV
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        # Обновляем модель
        success = updater.update_model(csv_path)
        assert success is True

        # Проверяем, что модель обучена
        assert updater.predictor.model is not None
        assert updater.last_update is not None

    def test_update_model_without_data(self, updater):
        """Тест обновления без данных."""
        success = updater.update_model()
        assert success is False

    def test_update_model_with_existing_data(self, updater, sample_data, temp_dir):
        """Тест обновления с уже существующими данными."""
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        # Первое обновление
        success1 = updater.update_model(csv_path)
        assert success1 is True

        # Создаем новые данные
        new_data = sample_data.copy()
        new_data["price"] = new_data["price"] + 10
        new_csv_path = os.path.join(temp_dir, "new_data.csv")
        new_data.to_csv(new_csv_path, index=False)

        # Второе обновление
        success2 = updater.update_model(new_csv_path)
        assert success2 is True

        # Проверяем, что данные объединены
        db_data = updater.predictor.db_manager.get_all_data()
        assert len(db_data) >= len(sample_data)

    def test_schedule_updates(self, updater):
        """Тест планирования обновлений."""
        updater.schedule_updates(60)
        assert updater.update_interval == 60

    def test_run_periodic_updates(self, updater):
        """Тест запуска периодических обновлений."""
        # Устанавливаем очень маленький интервал для теста
        updater.update_interval = 1

        # Запускаем в отдельном потоке
        updater.run_periodic_updates(run_immediately=False)

        assert updater.is_running is True

        # Останавливаем
        updater.stop_periodic_updates()
        assert updater.is_running is False

    def test_get_update_status(self, updater):
        """Тест получения статуса обновлений."""
        status = updater.get_update_status()

        assert "is_running" in status
        assert status["is_running"] is False
        assert "last_update" in status
        assert "update_interval" in status
        assert "data_path" in status

    def test_get_update_status_after_update(self, updater, sample_data, temp_dir):
        """Тест статуса после обновления."""
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        updater.update_model(csv_path)

        status = updater.get_update_status()
        assert status["last_update"] is not None
        assert "seconds_since_update" in status
        assert "next_update_in" in status

    def test_manual_update(self, updater, sample_data, temp_dir):
        """Тест ручного обновления."""
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        result = updater.manual_update(csv_path)

        assert "success" in result
        assert result["success"] is True
        assert "elapsed_time" in result
        assert "timestamp" in result

    def test_manual_update_fails(self, updater):
        """Тест неудачного ручного обновления."""
        result = updater.manual_update()

        assert "success" in result
        assert result["success"] is False

    def test_stop_periodic_updates(self, updater):
        """Тест остановки периодических обновлений."""
        updater.is_running = True
        updater.stop_periodic_updates()
        assert updater.is_running is False
