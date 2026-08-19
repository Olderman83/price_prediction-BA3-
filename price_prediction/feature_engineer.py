"""Модуль для создания признаков."""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    """
    Класс для создания и преобразования признаков.
    """

    def __init__(self):
        """Инициализация инженера признаков."""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self._shift_params = {}

    def create_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("Пустой DataFrame для создания признаков")
            return df

        self.logger.info("Начало создания признаков")
        df_features = df.copy()

        if "price" in df_features.columns:
            self.logger.warning("price найдена в данных! Удаляем...")
            df_features = df_features.drop(columns=["price"])

        # ТОЛЬКО БЕЗОПАСНЫЕ МЕТОДЫ
        df_features = self._create_basic_features(df_features, fit=fit)
        df_features = self._create_interaction_features(df_features, fit=fit)

        if "created_at" in df_features.columns:
            df_features = self._create_temporal_features(df_features, fit=fit)

        self.logger.info(f"Создано {len(df_features.columns)} признаков")
        self.feature_names = list(df_features.columns)

        return df_features

    def _create_basic_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Создание базовых признаков БЕЗ использования price."""
        df_features = df.copy()
        epsilon = 1e-10

        if fit:
            self._shift_params = {}
            for col in ["count", "add_cost"]:
                if col in df_features.columns:
                    min_val = df_features[col].min()
                    self._shift_params[col] = abs(min_val) + 1 if min_val <= 0 else 0

        for col in ["count", "add_cost"]:
            if col in df_features.columns:
                shift = self._shift_params.get(col, 1)
                if shift > 0:
                    df_features[f"log_{col}"] = np.log1p(df_features[col] + shift)
                else:
                    df_features[f"log_{col}"] = np.log1p(df_features[col])

        if "count" in df_features.columns and "add_cost" in df_features.columns:
            df_features["cost_per_count"] = df_features["add_cost"] / (
                df_features["count"] + epsilon
            )

        return df_features

    def _create_interaction_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Создание признаков взаимодействия БЕЗ использования price."""
        df_features = df.copy()
        numeric_cols = ["count", "add_cost"]
        present_cols = [col for col in numeric_cols if col in df_features.columns]

        for i in range(len(present_cols)):
            for j in range(i + 1, len(present_cols)):
                col1 = present_cols[i]
                col2 = present_cols[j]
                df_features[f"{col1}_x_{col2}"] = df_features[col1] * df_features[col2]
                df_features[f"{col1}_x_{col2}"] = df_features[
                    f"{col1}_x_{col2}"
                ].replace([np.inf, -np.inf], np.nan)

        return df_features

    def _create_temporal_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Создание временных признаков."""
        df_features = df.copy()

        if "created_at" in df_features.columns:
            df_features["created_at"] = pd.to_datetime(df_features["created_at"])
            df_features["year"] = df_features["created_at"].dt.year
            df_features["month"] = df_features["created_at"].dt.month
            df_features["day"] = df_features["created_at"].dt.day
            df_features["dayofweek"] = df_features["created_at"].dt.dayofweek
            df_features["quarter"] = df_features["created_at"].dt.quarter
            df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
            df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
            df_features["dayofweek_sin"] = np.sin(
                2 * np.pi * df_features["dayofweek"] / 7
            )
            df_features["dayofweek_cos"] = np.cos(
                2 * np.pi * df_features["dayofweek"] / 7
            )

        return df_features

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Кодирование категориальных переменных."""
        df_encoded = df.copy()
        categorical_cols = ["company", "product"]

        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = df_encoded[col].astype(str)
                    self.label_encoders[col].fit(df_encoded[col])
                    df_encoded[f"{col}_encoded"] = self.label_encoders[col].transform(
                        df_encoded[col]
                    )
                else:
                    if col in self.label_encoders:
                        df_encoded[col] = df_encoded[col].astype(str)
                        known_categories = set(self.label_encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_categories else "unknown"
                        )
                        if "unknown" not in self.label_encoders[col].classes_:
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, "unknown"
                            )
                        df_encoded[f"{col}_encoded"] = self.label_encoders[
                            col
                        ].transform(df_encoded[col])
                    else:
                        self.logger.warning(
                            f"Кодировщик для {col} не найден, пропускаем"
                        )

        return df_encoded

    def scale_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Масштабирование признаков (перезаписывает оригиналы)."""
        df_scaled = df.copy()

        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_cols if col != "price"]

        if not feature_columns:
            self.logger.warning("Нет признаков для масштабирования")
            return df_scaled

        if fit:
            self.scaler.fit(df_scaled[feature_columns])

        scaled_values = self.scaler.transform(df_scaled[feature_columns])

        for i, col in enumerate(feature_columns):
            df_scaled[col] = scaled_values[:, i]  # Перезаписываем оригиналы

        return df_scaled

    def prepare_for_model(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Полная подготовка данных для модели."""
        df_features = self.create_features(df, fit=fit)
        df_encoded = self.encode_categorical(df_features, fit=fit)
        df_scaled = self.scale_features(df_encoded, fit=fit)
        feature_columns = self._select_model_features(df_scaled)
        return df_scaled[feature_columns], feature_columns

    def prepare_for_prediction(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """Подготовка данных для прогнозирования."""
        df_features = self.create_features(df, fit=fit)
        df_encoded = self.encode_categorical(df_features, fit=fit)
        df_scaled = self.scale_features(df_encoded, fit=fit)
        feature_columns = self._select_model_features(df_scaled)
        return df_scaled[feature_columns]

    def _select_model_features(self, df: pd.DataFrame) -> List[str]:
        """
        Выбор признаков для модели - ТОЛЬКО БЕЗОПАСНЫЕ!
        """
        # Безопасные признаки (НЕТ агрегаций!)
        safe_features = [
            "count",  # исходный
            "add_cost",  # исходный
            "log_count",  # логарифм
            "log_add_cost",  # логарифм
            "cost_per_count",  # отношение
            "count_x_add_cost",  # взаимодействие
            "company_encoded",  # закодированная категория
            "product_encoded",  # закодированная категория
            "year",  # временной
            "month",  # временной
            "day",  # временной
            "dayofweek",  # временной
            "quarter",  # временной
            "month_sin",  # циклический
            "month_cos",  # циклический
            "dayofweek_sin",  # циклический
            "dayofweek_cos",  # циклический
        ]

        # Берём только те, которые есть в данных
        feature_cols = [col for col in safe_features if col in df.columns]

        self.logger.info(f"Выбрано {len(feature_cols)} безопасных признаков")
        return feature_cols

    def save_encoders(self, path: str) -> None:
        """Сохранение LabelEncoders."""
        import pickle

        encoders_path = os.path.join(path, "label_encoders.pkl")
        with open(encoders_path, "wb") as f:
            pickle.dump(self.label_encoders, f)

    def load_encoders(self, path: str) -> None:
        """Загрузка LabelEncoders."""
        import pickle

        encoders_path = os.path.join(path, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                self.label_encoders = pickle.load(f)
                self.logger.info("LabelEncoders загружены")
