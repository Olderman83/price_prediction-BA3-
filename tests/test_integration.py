"""Интеграционные тесты."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from price_prediction import (DatabaseManager, DataLoader, FeatureEngineer,
                              ModelUpdater, PricePredictor)


class TestIntegration:
    """Интеграционные тесты."""

    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных."""
        np.random.seed(42)
        n = 300
        data = pd.DataFrame(
            {
                "price": 100 + np.random.randn(n) * 50,
                "count": np.random.randint(10, 100, n),
                "add_cost": np.random.randint(100, 1000, n),
                "company": np.random.choice(["A", "B", "C", "D"], n),
                "product": np.random.choice(["X", "Y", "Z"], n),
                "created_at": pd.date_range("2023-01-01", periods=n),
            }
        )
        # Добавляем корреляцию
        data["price"] = (
            data["price"]
            + data["count"] * 0.5
            + data["add_cost"] * 0.1
            + np.random.randn(n) * 10
        )
        return data

    @pytest.fixture
    def temp_dir(self):
        """Создание временной директории."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_full_pipeline(self, sample_data, temp_dir):
        """Тест полного пайплайна."""
        db_path = os.path.join(temp_dir, "test.db")
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # 1. Сохраняем данные в CSV
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        # 2. Создаем предиктор
        predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir=model_dir
        )

        # 3. Загружаем и подготавливаем данные
        loader = DataLoader(csv_path)
        df = loader.load_data()
        df_processed = loader.preprocess_data(df)

        # 4. Сохраняем в БД
        predictor.db_manager.insert_batch(df_processed)

        # 5. Обучаем модель
        metrics = predictor.train_model()

        # 6. Проверяем метрики
        assert "r2" in metrics
        assert metrics["r2"] > 0.5  # Должна быть хотя бы небольшая корреляция

        # 7. Делаем прогноз
        features = {"company": "A", "product": "X", "count": 50, "add_cost": 500}
        price = predictor.predict(features)
        assert price > 0

        # 8. Загружаем модель заново
        new_predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir=model_dir
        )
        new_predictor._load_model()

        # 9. Проверяем, что прогнозы совпадают
        new_price = new_predictor.predict(features)
        assert abs(price - new_price) < 1e-6

        # 10. Получаем информацию о модели
        info = new_predictor.get_model_info()
        assert info["is_trained"] is True
        assert info["model_type"] == "ridge"

    def test_feature_engineer_integration(self, sample_data, temp_dir):
        """Тест интеграции FeatureEngineer."""
        engineer = FeatureEngineer()

        # Создаем признаки
        data_no_price = sample_data.drop(columns=["price"])
        df_features = engineer.create_features(data_no_price, fit=True)

        # Кодируем категории
        df_encoded = engineer.encode_categorical(df_features, fit=True)

        # Масштабируем
        df_scaled = engineer.scale_features(df_encoded, fit=True)

        # Выбираем признаки для модели
        feature_cols = engineer._select_model_features(df_scaled)

        assert len(feature_cols) > 0
        assert all(col in df_scaled.columns for col in feature_cols)

        # Подготавливаем для модели
        X, cols = engineer.prepare_for_model(sample_data, fit=True)
        assert len(X) == len(sample_data)
        assert len(cols) > 0

    def test_data_loader_and_database_integration(self, sample_data, temp_dir):
        """Тест интеграции DataLoader и DatabaseManager."""
        db_path = os.path.join(temp_dir, "test.db")

        # 1. Сохраняем данные в CSV
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        # 2. Загружаем через DataLoader
        loader = DataLoader(csv_path)
        df = loader.load_data()
        df_processed = loader.preprocess_data(df)

        # 3. Сохраняем в БД
        db_manager = DatabaseManager(db_path)
        count = db_manager.insert_batch(df_processed)
        assert count == len(df_processed)

        # 4. Получаем из БД
        df_from_db = db_manager.get_all_data()
        assert len(df_from_db) == len(df_processed)

        # 5. Проверяем статистику
        stats = db_manager.get_statistics()
        assert stats["total_records"] == len(df_processed)

    def test_model_updater_integration(self, sample_data, temp_dir):
        """Тест интеграции ModelUpdater."""
        db_path = os.path.join(temp_dir, "test.db")
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # 1. Создаем предиктор
        predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir=model_dir
        )

        # 2. Создаем обновлятор
        updater = ModelUpdater(predictor=predictor)

        # 3. Сохраняем данные в CSV
        csv_path = os.path.join(temp_dir, "data.csv")
        sample_data.to_csv(csv_path, index=False)

        # 4. Обновляем модель
        result = updater.manual_update(csv_path)
        assert result["success"] is True

        # 5. Проверяем, что модель обучена
        assert predictor.model is not None

        # 6. Добавляем новые данные
        new_data = sample_data.copy()
        new_data["price"] = new_data["price"] + 20
        new_csv_path = os.path.join(temp_dir, "new_data.csv")
        new_data.to_csv(new_csv_path, index=False)

        # 7. Обновляем снова
        result2 = updater.manual_update(new_csv_path)
        assert result2["success"] is True

        # 8. Проверяем статус
        status = updater.get_update_status()
        assert status["last_update"] is not None

    def test_prediction_with_context(self, sample_data, temp_dir):
        """Тест прогнозирования с контекстом."""
        db_path = os.path.join(temp_dir, "test.db")
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir=model_dir
        )

        # Обучаем модель
        predictor.train_model(data=sample_data)

        # Прогноз с контекстом
        features = {"company": "A", "product": "X", "count": 50, "add_cost": 500}

        price_no_context = predictor.predict(features, use_latest_data=False)
        price_with_context = predictor.predict(features, use_latest_data=True)

        # Оба прогноза должны быть валидными
        assert price_no_context > 0
        assert price_with_context > 0

    def test_batch_prediction(self, sample_data, temp_dir):
        """Тест пакетного прогнозирования."""
        db_path = os.path.join(temp_dir, "test.db")
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir=model_dir
        )

        # Обучаем модель
        predictor.train_model(data=sample_data)

        # Несколько прогнозов
        test_cases = [
            {"company": "A", "product": "X", "count": 50, "add_cost": 500},
            {"company": "B", "product": "Y", "count": 70, "add_cost": 700},
            {"company": "C", "product": "Z", "count": 30, "add_cost": 300},
        ]

        predictions = []
        for case in test_cases:
            price = predictor.predict(case)
            predictions.append(price)
            assert price > 0

        assert len(predictions) == 3
        assert all(isinstance(p, float) for p in predictions)

    def test_error_handling_integration(self, sample_data, temp_dir):
        """Тест обработки ошибок в интеграционном сценарии."""
        db_path = os.path.join(temp_dir, "test.db")
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir=model_dir
        )

        # Прогноз без обучения
        with pytest.raises(ValueError):
            predictor.predict(
                {"company": "A", "product": "X", "count": 50, "add_cost": 500}
            )

        # Обучаем
        predictor.train_model(data=sample_data)

        # Прогноз с отсутствующими данными
        with pytest.raises(KeyError):
            # Отсутствует 'add_cost'
            predictor.predict({"company": "A", "product": "X", "count": 50})

        # Загрузка модели из несуществующей директории
        bad_predictor = PricePredictor(
            model_type="ridge", db_path=db_path, model_dir="/nonexistent/directory"
        )
        success = bad_predictor._load_model()
        assert success is False
