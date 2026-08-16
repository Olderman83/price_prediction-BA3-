"""Модуль для создания признаков."""
import os
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


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

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание всех признаков.

        Args:
            df: Исходные данные

        Returns:
            pd.DataFrame: Данные с созданными признаками
        """
        if df.empty:
            self.logger.warning("Пустой DataFrame для создания признаков")
            return df

        self.logger.info("Начало создания признаков")
        df_features = df.copy()

        # Базовые признаки
        df_features = self._create_basic_features(df_features)

        # Признаки взаимодействия
        df_features = self._create_interaction_features(df_features)

        # Агрегированные признаки
        df_features = self._create_aggregation_features(df_features)

        # Временные признаки (если есть дата)
        if 'created_at' in df_features.columns:
            df_features = self._create_temporal_features(df_features)

        self.logger.info(f"Создано {len(df_features.columns)} признаков")
        self.feature_names = list(df_features.columns)

        return df_features

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание базовых признаков.

        Args:
            df: DataFrame для обработки

        Returns:
            pd.DataFrame: DataFrame с базовыми признаками
        """
        df_features = df.copy()
        epsilon = 1e-10  # Маленькое число для защиты от деления на ноль

        # Логарифмические преобразования
        for col in ['price', 'count', 'add_cost']:
            if col in df_features.columns:
                min_val = df_features[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    df_features[f'log_{col}'] = np.log1p(df_features[col] + shift)
                else:
                    df_features[f'log_{col}'] = np.log1p(df_features[col])

        # Отношения признаков (защита от деления на ноль)
        if 'price' in df_features.columns and 'count' in df_features.columns:
            df_features['price_per_count'] = df_features['price'] / (df_features['count'] + epsilon)

        if 'price' in df_features.columns and 'add_cost' in df_features.columns:
            df_features['price_per_cost'] = df_features['price'] / (df_features['add_cost'] + epsilon)

        if 'count' in df_features.columns and 'add_cost' in df_features.columns:
            df_features['cost_per_count'] = df_features['add_cost'] / (df_features['count'] + epsilon)

        return df_features

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков взаимодействия.

        Args:
            df: DataFrame для обработки

        Returns:
            pd.DataFrame: DataFrame с признаками взаимодействия
        """
        df_features = df.copy()

        # Комбинации числовых признаков
        numeric_cols = ['price', 'count', 'add_cost']
        present_cols = [col for col in numeric_cols if col in df_features.columns]

        for i in range(len(present_cols)):
            for j in range(i + 1, len(present_cols)):
                col1 = present_cols[i]
                col2 = present_cols[j]
                df_features[f'{col1}_x_{col2}'] = df_features[col1] * df_features[col2]
                df_features[f'{col1}_x_{col2}'] = df_features[f'{col1}_x_{col2}'].replace([np.inf, -np.inf], np.nan)

        return df_features

    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание агрегированных признаков по группам.

        Args:
            df: DataFrame для обработки

        Returns:
            pd.DataFrame: DataFrame с агрегированными признаками
        """
        df_features = df.copy()

        # Агрегации по компании
        if 'company' in df_features.columns and 'price' in df_features.columns:
            company_stats = df_features.groupby('company')['price'].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).add_prefix('company_price_')
            df_features = df_features.merge(company_stats, on='company', how='left')

        # Агрегации по продукту
        if 'product' in df_features.columns and 'price' in df_features.columns:
            product_stats = df_features.groupby('product')['price'].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).add_prefix('product_price_')
            df_features = df_features.merge(product_stats, on='product', how='left')

        # Агрегации по компании и продукту
        if 'company' in df_features.columns and 'product' in df_features.columns:
            df_features['company_product'] = df_features['company'] + '_' + df_features['product']

        return df_features

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание временных признаков.

        Args:
            df: DataFrame для обработки

        Returns:
            pd.DataFrame: DataFrame с временными признаками
        """
        df_features = df.copy()

        if 'created_at' in df_features.columns:
            df_features['created_at'] = pd.to_datetime(df_features['created_at'])

            # Извлечение компонентов даты
            df_features['year'] = df_features['created_at'].dt.year
            df_features['month'] = df_features['created_at'].dt.month
            df_features['day'] = df_features['created_at'].dt.day
            df_features['dayofweek'] = df_features['created_at'].dt.dayofweek
            df_features['quarter'] = df_features['created_at'].dt.quarter

            # Циклическое кодирование для временных признаков
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
            df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)

        return df_features

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Кодирование категориальных переменных.

        Args:
            df: DataFrame для кодирования
            fit: True для обучения кодировщиков, False для применения существующих

        Returns:
            pd.DataFrame: DataFrame с закодированными переменными
        """
        df_encoded = df.copy()
        categorical_cols = ['company', 'product']

        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    # Обработка неизвестных категорий
                    df_encoded[col] = df_encoded[col].astype(str)
                    self.label_encoders[col].fit(df_encoded[col])
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
                else:
                    if col in self.label_encoders:
                        # Преобразование с обработкой неизвестных категорий
                        df_encoded[col] = df_encoded[col].astype(str)
                        # Замена неизвестных категорий на 'unknown'
                        known_categories = set(self.label_encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_categories else 'unknown'
                        )
                        # Добавление 'unknown' в кодировщик
                        if 'unknown' not in self.label_encoders[col].classes_:
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, 'unknown'
                            )
                        df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(
                            df_encoded[col]
                        )
                    else:
                        self.logger.warning(f"Кодировщик для {col} не найден, пропускаем")

        return df_encoded

    def scale_features(
            self,
            df: pd.DataFrame,
            fit: bool = True,
            feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Масштабирование признаков.

        Args:
            df: DataFrame для масштабирования
            fit: True для обучения скалера
            feature_columns: Список колонок для масштабирования

        Returns:
            pd.DataFrame: DataFrame с масштабированными признаками
        """
        df_scaled = df.copy()

        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Исключаем целевой признак
            feature_columns = [col for col in numeric_cols if col != 'price']

        if not feature_columns:
            self.logger.warning("Нет признаков для масштабирования")
            return df_scaled

        if fit:
            self.scaler.fit(df_scaled[feature_columns])

        scaled_values = self.scaler.transform(df_scaled[feature_columns])

        for i, col in enumerate(feature_columns):
            df_scaled[f'{col}_scaled'] = scaled_values[:, i]

        return df_scaled

    def prepare_for_model(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Полная подготовка данных для модели.

        Args:
            df: Исходные данные
            fit: True для обучения трансформеров

        Returns:
            Tuple[pd.DataFrame, List[str]]: Подготовленные данные и список признаков
        """
        # Создание признаков
        df_features = self.create_features(df)

        # Кодирование категориальных переменных
        df_encoded = self.encode_categorical(df_features, fit=fit)

        # Масштабирование
        df_scaled = self.scale_features(df_encoded, fit=fit)

        # Выбор признаков для модели
        feature_columns = self._select_model_features(df_scaled)

        return df_scaled[feature_columns], feature_columns

    def _select_model_features(self, df: pd.DataFrame) -> List[str]:
        """
        Выбор признаков для модели.

        Args:
            df: DataFrame с признаками

        Returns:
            List[str]: Список выбранных признаков
        """
        # Исключаем целевой признак и временные колонки
        exclude_cols = ['price', 'created_at', 'company', 'product']

        # Добавляем все числовые колонки, кроме исключенных
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Если признаков слишком много, выбираем наиболее важные
        if len(feature_cols) > 50:
            # В реальном проекте здесь можно добавить выбор признаков
            self.logger.info(f"Выбрано {len(feature_cols)} признаков")

        return feature_cols

    def save_encoders(self, path: str) -> None:
        """Сохранение LabelEncoders."""
        import pickle
        encoders_path = os.path.join(path, 'label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)

    def load_encoders(self, path: str) -> None:
        """Загрузка LabelEncoders."""
        import pickle
        encoders_path = os.path.join(path, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
                self.logger.info("LabelEncoders загружены")
