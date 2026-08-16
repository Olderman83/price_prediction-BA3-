"""Модуль для загрузки и предобработки данных."""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
import os


class DataLoader:
    """
    Класс для загрузки и предобработки данных из CSV.
    """

    REQUIRED_COLUMNS = ['price', 'count', 'add_cost', 'company', 'product']

    def __init__(self, filepath: Optional[str] = None):
        """
        Инициализация загрузчика данных.

        Args:
            filepath: Путь к CSV файлу (опционально)
        """
        self.filepath = filepath
        self.data = None
        self.logger = logging.getLogger(__name__)

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.

        Args:
            filepath: Путь к CSV файлу

        Returns:
            pd.DataFrame: Загруженные данные
        """
        if filepath:
            self.filepath = filepath

        if not self.filepath:
            raise ValueError("Не указан путь к файлу")

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Файл не найден: {self.filepath}")

        try:
            self.logger.info(f"Загрузка данных из {self.filepath}")
            self.data = pd.read_csv(self.filepath)
            self.logger.info(f"Загружено {len(self.data)} записей")
            return self.data
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Предобработка данных.

        Args:
            df: DataFrame для обработки (опционально)

        Returns:
            pd.DataFrame: Обработанные данные
        """
        if df is None:
            df = self.data

        if df is None:
            raise ValueError("Нет данных для предобработки")

        self.logger.info("Начало предобработки данных")

        # Проверка наличия необходимых колонок
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")

        # Создаем копию данных
        df_processed = df.copy()

        # Удаление дубликатов
        initial_len = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        if len(df_processed) < initial_len:
            self.logger.info(f"Удалено {initial_len - len(df_processed)} дубликатов")

        # Обработка пропущенных значений
        self._handle_missing_values(df_processed)

        # Обработка выбросов
        self._handle_outliers(df_processed)

        # Нормализация числовых колонок
        self._normalize_numerical(df_processed)

        # Кодирование категориальных переменных
        df_processed = self._encode_categorical(df_processed)

        self.logger.info(f"Предобработка завершена. Размер данных: {len(df_processed)}")
        return df_processed

    def _handle_missing_values(self, df: pd.DataFrame) -> None:
        """
        Обработка пропущенных значений.

        Args:
            df: DataFrame для обработки
        """
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            self.logger.info(f"Обнаружены пропущенные значения:\n{missing_values[missing_values > 0]}")

            # Для числовых колонок - заполняем медианой
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    self.logger.info(f"Заполнены пропуски в {col} медианой: {median_val}")

            # Для категориальных - заполняем модой
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    self.logger.info(f"Заполнены пропуски в {col} модой: {mode_val}")

    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> None:
        """
        Обработка выбросов.

        Args:
            df: DataFrame для обработки
            method: Метод обработки ('iqr' или 'zscore')
        """
        numeric_cols = ['price', 'count', 'add_cost']

        for col in numeric_cols:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                if outliers.sum() > 0:
                    self.logger.info(f"Найдено {outliers.sum()} выбросов в {col}")
                    # Заменяем выбросы на границы
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound

            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outliers = z_scores > 3
                    if outliers.sum() > 0:
                        self.logger.info(f"Найдено {outliers.sum()} выбросов в {col}")
                        df.loc[outliers, col] = mean

    def _normalize_numerical(self, df: pd.DataFrame) -> None:
        """
        Нормализация числовых колонок.

        Args:
            df: DataFrame для обработки
        """
        numeric_cols = ['price', 'count', 'add_cost']
        for col in numeric_cols:
            if col in df.columns:
                # Минимаксная нормализация
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_normalized'] = 0

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирование категориальных переменных.

        Args:
            df: DataFrame для обработки

        Returns:
            pd.DataFrame: DataFrame с закодированными переменными
        """
        categorical_cols = ['company', 'product']
        df_encoded = df.copy()

        for col in categorical_cols:
            if col in df_encoded.columns:
                # One-hot encoding для категориальных переменных
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                # Удаляем оригинальную колонку
                df_encoded.drop(columns=[col], inplace=True)

        return df_encoded

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Получение сводной статистики по данным.

        Returns:
            Dict[str, Any]: Словарь со статистиками
        """
        if self.data is None:
            return {}

        numeric_cols = ['price', 'count', 'add_cost']
        stats = {
            'total_rows': len(self.data),
            'columns': list(self.data.columns),
            'numeric_stats': {}
        }

        for col in numeric_cols:
            if col in self.data.columns:
                stats['numeric_stats'][col] = {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max()),
                    'q25': float(self.data[col].quantile(0.25)),
                    'q50': float(self.data[col].quantile(0.50)),
                    'q75': float(self.data[col].quantile(0.75))
                }

        stats['categorical_stats'] = {}
        for col in ['company', 'product']:
            if col in self.data.columns:
                stats['categorical_stats'][col] = {
                    'unique_count': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts().head(5).to_dict()
                }

        return stats
