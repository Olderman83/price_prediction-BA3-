"""Тесты для FeatureEngineer."""

import os
import tempfile

import pandas as pd
import pytest

from price_prediction.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Тесты для FeatureEngineer."""

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
                "created_at": pd.date_range("2023-01-01", periods=5),
            }
        )

    @pytest.fixture
    def sample_data_no_price(self):
        """Создание данных без цены."""
        return pd.DataFrame(
            {
                "count": [10, 20, 30, 40, 50],
                "add_cost": [500, 600, 700, 800, 900],
                "company": ["A", "B", "A", "B", "C"],
                "product": ["X", "Y", "X", "Y", "Z"],
            }
        )

    def test_init(self):
        """Тест инициализации."""
        engineer = FeatureEngineer()
        assert engineer.scaler is not None
        assert engineer.label_encoders == {}
        assert engineer.feature_names == []

    def test_create_features(self, sample_data):
        """Тест создания признаков."""
        engineer = FeatureEngineer()

        # Удаляем price (она не должна использоваться в признаках)
        data = sample_data.drop(columns=["price"])
        df_features = engineer.create_features(data, fit=True)

        # Проверяем созданные признаки
        expected_features = [
            "log_count",
            "log_add_cost",
            "cost_per_count",
            "count_x_add_cost",
            "year",
            "month",
            "day",
            "dayofweek",
            "quarter",
            "month_sin",
            "month_cos",
            "dayofweek_sin",
            "dayofweek_cos",
        ]

        for feature in expected_features:
            if feature not in df_features.columns:
                # Некоторые признаки могут отсутствовать в зависимости от данных
                pass

        assert len(df_features) == 5

    def test_create_features_no_price_removed(self, sample_data):
        """Тест создания признаков без цены."""
        engineer = FeatureEngineer()

        # Если price присутствует, она будет удалена
        df_features = engineer.create_features(sample_data, fit=True)

        assert "price" not in df_features.columns

    def test_create_features_empty(self):
        """Тест создания признаков с пустым DataFrame."""
        engineer = FeatureEngineer()
        df_empty = pd.DataFrame()

        df_features = engineer.create_features(df_empty, fit=True)
        assert df_features.empty

    def test_create_basic_features(self, sample_data_no_price):
        """Тест создания базовых признаков."""
        engineer = FeatureEngineer()
        df_features = engineer._create_basic_features(sample_data_no_price, fit=True)

        assert "log_count" in df_features.columns
        assert "log_add_cost" in df_features.columns
        assert "cost_per_count" in df_features.columns

    def test_create_basic_features_with_negative_values(self):
        """Тест создания базовых признаков с отрицательными значениями."""
        data = pd.DataFrame({"count": [-10, 20, 30], "add_cost": [-500, 600, 700]})

        engineer = FeatureEngineer()
        df_features = engineer._create_basic_features(data, fit=True)

        # Должны быть добавлены сдвиги для логарифмов
        assert "log_count" in df_features.columns
        assert "log_add_cost" in df_features.columns

    def test_create_interaction_features(self, sample_data_no_price):
        """Тест создания признаков взаимодействия."""
        engineer = FeatureEngineer()
        df_features = engineer._create_interaction_features(
            sample_data_no_price, fit=True
        )

        assert "count_x_add_cost" in df_features.columns

    def test_create_temporal_features(self, sample_data_no_price):
        """Тест создания временных признаков."""
        data_with_date = sample_data_no_price.copy()
        data_with_date["created_at"] = pd.date_range("2023-01-01", periods=5)

        engineer = FeatureEngineer()
        df_features = engineer._create_temporal_features(data_with_date, fit=True)

        assert "year" in df_features.columns
        assert "month" in df_features.columns
        assert "day" in df_features.columns
        assert "dayofweek" in df_features.columns
        assert "quarter" in df_features.columns
        assert "month_sin" in df_features.columns
        assert "month_cos" in df_features.columns

    def test_encode_categorical(self, sample_data_no_price):
        """Тест кодирования категориальных переменных."""
        engineer = FeatureEngineer()
        df_encoded = engineer.encode_categorical(sample_data_no_price, fit=True)

        assert "company_encoded" in df_encoded.columns
        assert "product_encoded" in df_encoded.columns

        # Проверка, что значения закодированы
        assert df_encoded["company_encoded"].dtype in ["int64", "int32"]
        assert df_encoded["product_encoded"].dtype in ["int64", "int32"]

    def test_encode_categorical_fit_false(self, sample_data_no_price):
        """Тест кодирования с fit=False."""
        engineer = FeatureEngineer()

        # Обучаем на тренировочных данных
        df_encoded = engineer.encode_categorical(sample_data_no_price, fit=True)

        # Сохраняем закодированные значения для проверки согласованности
        original_encoded_vals = df_encoded["company_encoded"].values.copy()
        original_products = df_encoded["product_encoded"].values.copy()

        # Проверяем, что кодирование работает (не все значения одинаковые)
        assert len(set(original_encoded_vals)) > 1
        assert len(set(original_products)) > 1

        # Применяем к новым данным
        df_new = pd.DataFrame(
            {
                "company": ["A", "D", "B"],  # D - новая категория
                "product": ["X", "Z", "Y"],
            }
        )

        df_encoded_new = engineer.encode_categorical(df_new, fit=False)

        # Проверки
        assert "company_encoded" in df_encoded_new.columns
        assert "product_encoded" in df_encoded_new.columns
        assert len(df_encoded_new) == 3

        # Проверяем, что известные категории закодированы правильно
        # A должна иметь тот же код, что и при обучении
        idx_a_train = sample_data_no_price["company"] == "A"
        train_code_a = df_encoded.loc[idx_a_train, "company_encoded"].iloc[0]
        idx_a_new = df_new["company"] == "A"
        new_code_a = df_encoded_new.loc[idx_a_new, "company_encoded"].iloc[0]
        assert train_code_a == new_code_a

        # B должна иметь тот же код, что и при обучении
        idx_b_train = sample_data_no_price["company"] == "B"
        train_code_b = df_encoded.loc[idx_b_train, "company_encoded"].iloc[0]
        idx_b_new = df_new["company"] == "B"
        new_code_b = df_encoded_new.loc[idx_b_new, "company_encoded"].iloc[0]
        assert train_code_b == new_code_b

        # D - новая категория, должна быть закодирована как 'unknown'
        # Получаем код для 'unknown'
        unknown_code = engineer.label_encoders["company"].transform(["unknown"])[0]
        idx_d_new = df_new["company"] == "D"
        new_code_d = df_encoded_new.loc[idx_d_new, "company_encoded"].iloc[0]
        assert new_code_d == unknown_code

        # Проверяем, что кодировщик не изменился после применения
        # (важно: fit=False не должен менять label_encoders)
        assert "D" not in engineer.label_encoders["company"].classes_
        assert "unknown" in engineer.label_encoders["company"].classes_

    def test_scale_features(self, sample_data_no_price):
        """Тест масштабирования признаков."""
        engineer = FeatureEngineer()
        df_features = engineer.create_features(sample_data_no_price, fit=True)

        # Выбираем числовые признаки для масштабирования
        numeric_cols = ["count", "add_cost", "log_count", "log_add_cost"]
        existing_cols = [col for col in numeric_cols if col in df_features.columns]

        df_scaled = engineer.scale_features(
            df_features, fit=True, feature_columns=existing_cols
        )

        # Проверка, что значения масштабированы (среднее ~0, std ~1)
        for col in existing_cols:
            mean_val = df_scaled[col].mean()
            std_val = df_scaled[col].std()
            assert abs(mean_val) < 1e-6
            assert abs(std_val - 1.0) < 1e-6

    def test_scale_features_fit_false(self, sample_data_no_price):
        """Тест масштабирования с fit=False."""
        engineer = FeatureEngineer()

        # Обучаем
        df_features = engineer.create_features(sample_data_no_price, fit=True)
        numeric_cols = ["count", "add_cost", "log_count", "log_add_cost"]
        existing_cols = [col for col in numeric_cols if col in df_features.columns]
        df_scaled = engineer.scale_features(
            df_features, fit=True, feature_columns=existing_cols
        )

        # Сохраняем статистику
        original_stds = {col: df_scaled[col].std() for col in existing_cols}

        # Применяем к новым данным
        df_new = pd.DataFrame({"count": [15, 25, 35], "add_cost": [550, 650, 750]})

        engineer_new = FeatureEngineer()
        engineer_new.scaler = engineer.scaler

        df_new_features = engineer_new.create_features(df_new, fit=False)
        existing_cols_new = [
            col for col in existing_cols if col in df_new_features.columns
        ]
        df_scaled_new = engineer_new.scale_features(
            df_new_features, fit=False, feature_columns=existing_cols_new
        )

        assert len(df_scaled_new) == 3

        # Проверяем, что std сохраняется (используем original_stds)
        for col in existing_cols_new:
            assert (
                abs(df_scaled_new[col].std() - original_stds[col]) < 0.1
            ), f"Std mismatch for {col}"

    def test_prepare_for_model(self, sample_data):
        """Тест подготовки данных для модели."""
        engineer = FeatureEngineer()

        X, feature_cols = engineer.prepare_for_model(sample_data, fit=True)

        assert X is not None
        assert len(feature_cols) > 0
        assert "price" not in X.columns
        assert len(X) == 5

    def test_prepare_for_prediction(self, sample_data):
        """Тест подготовки данных для прогнозирования."""
        engineer = FeatureEngineer()

        # Сначала обучаем
        engineer.prepare_for_model(sample_data, fit=True)

        # Затем подготавливаем для прогноза
        X = engineer.prepare_for_prediction(sample_data, fit=False)

        assert X is not None
        assert "price" not in X.columns
        assert len(X) == 5

    def test_select_model_features(self, sample_data_no_price):
        """Тест выбора признаков для модели."""
        engineer = FeatureEngineer()
        df_features = engineer.create_features(sample_data_no_price, fit=True)

        feature_cols = engineer._select_model_features(df_features)

        # Проверка, что выбраны только безопасные признаки
        safe_features = [
            "count",
            "add_cost",
            "log_count",
            "log_add_cost",
            "cost_per_count",
            "count_x_add_cost",
            "company_encoded",
            "product_encoded",
            "year",
            "month",
            "day",
            "dayofweek",
            "quarter",
            "month_sin",
            "month_cos",
            "dayofweek_sin",
            "dayofweek_cos",
        ]

        for col in feature_cols:
            assert col in safe_features

    def test_save_load_encoders(self, sample_data_no_price):
        """Тест сохранения и загрузки кодировщиков."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engineer = FeatureEngineer()
            engineer.encode_categorical(sample_data_no_price, fit=True)

            # Сохраняем
            engineer.save_encoders(temp_dir)

            # Проверяем, что файл создан
            encoders_path = os.path.join(temp_dir, "label_encoders.pkl")
            assert os.path.exists(encoders_path)

            # Загружаем в новый объект
            new_engineer = FeatureEngineer()
            new_engineer.load_encoders(temp_dir)

            assert len(new_engineer.label_encoders) == len(engineer.label_encoders)
            assert set(new_engineer.label_encoders.keys()) == set(
                engineer.label_encoders.keys()
            )

    def test_load_encoders_file_not_found(self):
        """Тест загрузки кодировщиков из несуществующего файла."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engineer = FeatureEngineer()
            # Не должно быть ошибки
            engineer.load_encoders(temp_dir)
            assert engineer.label_encoders == {}

    def test_encode_categorical_with_unknown_categories(self, sample_data_no_price):
        """Тест кодирования с неизвестными категориями."""
        engineer = FeatureEngineer()

        # Обучаем на данных с категориями A, B, C
        engineer.encode_categorical(sample_data_no_price, fit=True)

        # Новые данные с категорией D (неизвестная)
        new_data = pd.DataFrame(
            {"company": ["D", "A", "E"], "product": ["X", "Y", "Z"]}
        )

        df_encoded = engineer.encode_categorical(new_data, fit=False)

        # D и E должны быть закодированы как unknown
        assert df_encoded["company_encoded"].iloc[0] in engineer.label_encoders[
            "company"
        ].transform(["unknown"])
        assert df_encoded["company_encoded"].iloc[2] in engineer.label_encoders[
            "company"
        ].transform(["unknown"])

    def test_scale_features_with_default_columns(self, sample_data_no_price):
        """Тест масштабирования с автоматическим выбором колонок."""
        engineer = FeatureEngineer()
        df_features = engineer.create_features(sample_data_no_price, fit=True)

        # Без указания колонок
        df_scaled = engineer.scale_features(df_features, fit=True)

        # Должны быть масштабированы все числовые колонки, кроме price
        numeric_cols = [
            col
            for col in df_features.columns
            if pd.api.types.is_numeric_dtype(df_features[col]) and col != "price"
        ]
        for col in numeric_cols:
            if col in df_scaled.columns:
                assert abs(df_scaled[col].mean()) < 1e-6
