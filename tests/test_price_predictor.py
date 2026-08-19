"""Тесты для PricePredictor."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from price_prediction.price_predictor import PricePredictor


class TestPricePredictor:
    """Тесты для PricePredictor."""

    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame(
            {
                "price": 100 + np.random.randn(n) * 50 + np.random.randn(n) * 20,
                "count": np.random.randint(10, 100, n),
                "add_cost": np.random.randint(100, 1000, n),
                "company": np.random.choice(["A", "B", "C", "D"], n),
                "product": np.random.choice(["X", "Y", "Z"], n),
                "created_at": pd.date_range("2023-01-01", periods=n),
            }
        )
        # Добавляем корреляцию цены с признаками
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

    @pytest.fixture
    def temp_db_path(self):
        """Создание временной БД."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def predictor(self, temp_db_path, temp_dir):
        """Создание предиктора."""
        return PricePredictor(
            model_type="ridge", db_path=temp_db_path, model_dir=temp_dir
        )

    def test_init(self, temp_db_path, temp_dir):
        """Тест инициализации."""
        predictor = PricePredictor(
            model_type="ridge", db_path=temp_db_path, model_dir=temp_dir
        )

        assert predictor.model_type == "ridge"
        assert predictor.model is None
        assert os.path.exists(temp_dir)

    def test_load_and_prepare_data(self, predictor, sample_data, temp_dir):
        """Тест загрузки и подготовки данных."""
        # Создаем временный CSV файл
        csv_path = os.path.join(temp_dir, "test_data.csv")
        sample_data.to_csv(csv_path, index=False)

        df_processed = predictor.load_and_prepare_data(csv_path)

        assert df_processed is not None
        assert len(df_processed) == len(sample_data)

        # Проверка, что данные сохранены в БД
        db_data = predictor.db_manager.get_all_data()
        assert len(db_data) > 0

    def test_train_model(self, predictor, sample_data):
        """Тест обучения модели."""
        # Подготовка данных
        predictor.db_manager.clear_data()
        predictor.db_manager.insert_batch(sample_data)

        metrics = predictor.train_model()

        assert metrics is not None
        assert "r2" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "train_r2" in metrics
        assert "test_r2" in metrics

        # Модель должна быть обучена
        assert predictor.model is not None
        assert len(predictor.feature_columns) > 0

    def test_train_model_with_data(self, predictor, sample_data):
        """Тест обучения с переданными данными."""
        metrics = predictor.train_model(data=sample_data)

        assert metrics is not None
        assert "r2" in metrics
        assert predictor.model is not None

    def test_train_model_empty_data(self, predictor):
        """Тест обучения с пустыми данными."""
        with pytest.raises(ValueError):
            predictor.train_model(data=pd.DataFrame())

    def test_predict(self, predictor, sample_data):
        """Тест прогнозирования."""
        # Обучаем модель
        predictor.train_model(data=sample_data)

        # Прогнозирование
        features = {"company": "A", "product": "X", "count": 50, "add_cost": 500}

        price = predictor.predict(features)
        assert isinstance(price, float)
        assert price > 0

    def test_predict_with_latest_data(self, predictor, sample_data):
        """Тест прогнозирования с использованием последних данных."""
        # Обучаем модель
        predictor.train_model(data=sample_data)

        features = {"company": "A", "product": "X", "count": 50, "add_cost": 500}

        price = predictor.predict(features, use_latest_data=True)
        assert isinstance(price, float)
        assert price > 0

    def test_predict_without_model(self, predictor):
        """Тест прогнозирования без обученной модели."""
        features = {"company": "A", "product": "X", "count": 50, "add_cost": 500}

        with pytest.raises(ValueError):
            predictor.predict(features)

    def test_evaluate(self, predictor, sample_data):
        """Тест оценки модели."""
        # Обучаем модель
        predictor.train_model(data=sample_data)

        # Оценка
        metrics = predictor.evaluate(sample_data)

        assert metrics is not None
        assert "r2" in metrics
        assert "mae" in metrics

    def test_evaluate_without_model(self, predictor, sample_data):
        """Тест оценки без обученной модели."""
        with pytest.raises(ValueError):
            predictor.evaluate(sample_data)

    def test_get_model_info(self, predictor, sample_data):
        """Тест получения информации о модели."""
        # До обучения
        info = predictor.get_model_info()
        assert info["is_trained"] is False

        # После обучения
        predictor.train_model(data=sample_data)
        info = predictor.get_model_info()
        assert info["is_trained"] is True
        assert "model_type" in info
        assert "feature_count" in info
        assert "features" in info

    def test_clean_data(self, predictor):
        """Тест очистки данных."""
        # Создаем данные с NaN
        X = pd.DataFrame(
            {"feature1": [1, 2, np.nan, 4, 5], "feature2": [1, np.nan, 3, 4, 5]}
        )
        y = np.array([1, 2, 3, 4, 5])

        # Обучение импьютера
        X_clean, y_clean = predictor._clean_data(X, y, fit=True)
        assert not X_clean.isnull().any().any()
        assert len(X_clean) == 5

        # Применение к новым данным
        X_new = pd.DataFrame({"feature1": [1, np.nan, 3], "feature2": [1, 2, 3]})
        y_new = np.array([1, 2, 3])

        X_clean2, y_clean2 = predictor._clean_data(X_new, y_new, fit=False)
        assert not X_clean2.isnull().any().any()

    def test_clean_data_all_nan(self, predictor):
        """Тест очистки данных, когда все данные NaN."""
        X = pd.DataFrame({"feature1": [np.nan, np.nan], "feature2": [np.nan, np.nan]})
        y = np.array([1, 2])

        with pytest.raises(ValueError):
            predictor._clean_data(X, y, fit=True)

    def test_create_model(self, predictor):
        """Тест создания модели."""
        # Ridge
        model = predictor._create_model()
        assert model is not None

        # Random Forest
        predictor.model_type = "random_forest"
        model = predictor._create_model()
        assert model is not None

        # Неподдерживаемый тип
        predictor.model_type = "unknown"
        with pytest.raises(ValueError):
            predictor._create_model()

    def test_create_context_features(self, predictor, sample_data):
        """Тест создания контекстных признаков."""
        context = predictor._create_context_features(sample_data)

        assert "count" in context
        assert "add_cost" in context
        assert "log_count" in context
        assert "log_add_cost" in context
        assert "cost_per_count" in context
        assert "count_x_add_cost" in context
        assert "year" in context
        assert "month" in context
        assert "day" in context
        assert "dayofweek" in context
        assert "quarter" in context
        assert "month_sin" in context
        assert "month_cos" in context

    def test_train_model_saves_model(self, predictor, sample_data, temp_dir):
        """Тест, что модель сохраняется после обучения."""
        predictor.train_model(data=sample_data)

        # Проверяем наличие файлов
        model_path = os.path.join(temp_dir, "model.pkl")
        feature_path = os.path.join(temp_dir, "features.pkl")

        assert os.path.exists(model_path)
        assert os.path.exists(feature_path)

    def test_load_model_without_file(self, predictor):
        """Тест загрузки модели без файла."""
        success = predictor._load_model()
        assert success is False

    def test_predict_with_feature_missing(self, predictor, sample_data):
        """Тест прогнозирования с отсутствующими признаками."""
        predictor.train_model(data=sample_data)

        features = {
            "company": "A",
            "product": "X",
            "count": 50,
            # add_cost отсутствует
        }

        # Должен работать без ошибки
        price = predictor.predict(features)
        assert isinstance(price, float)
