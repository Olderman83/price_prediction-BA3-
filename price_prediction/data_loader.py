import logging
import os
from typing import Any, Dict, Optional

import pandas as pd


class DataLoader:
    """
    Класс для загрузки и предобработки данных из CSV.
    """

    REQUIRED_COLUMNS = ["price", "count", "add_cost", "company", "product"]

    def __init__(self, filepath: Optional[str] = None):
        """
        Инициализация загрузчика данных.

        Args:
            filepath: Путь к CSV файлу (опционально)
        """
        self.filepath = filepath
        self.data = None
        self.logger = logging.getLogger(__name__)

        # Сохраняем параметры для предобработки
        self._preprocessing_params = {}

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла.
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

    def preprocess_data(
        self, df: Optional[pd.DataFrame] = None, fit: bool = True
    ) -> pd.DataFrame:
        """
        Предобработка данных (очистка, обработка пропусков, выбросов).

        Args:
            df: DataFrame для обработки
            fit: True для обучения параметров, False для применения сохраненных
        """
        if df is None:
            df = self.data

        if df is None:
            raise ValueError("Нет данных для предобработки")

        self.logger.info(f"Начало предобработки данных (fit={fit})")

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
        df_processed = self._handle_missing_values(df_processed, fit=fit)

        # Обработка выбросов
        df_processed = self._handle_outliers(df_processed, fit=fit)

        self.logger.info(f"Предобработка завершена. Размер данных: {len(df_processed)}")
        return df_processed

    def _handle_missing_values(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Обработка пропущенных значений.
        """
        df_clean = df.copy()
        missing_values = df_clean.isnull().sum()

        if missing_values.sum() == 0:
            return df_clean

        self.logger.info(
            f"Обнаружены пропущенные значения:\n{missing_values[missing_values > 0]}"
        )

        # Для числовых колонок
        numeric_cols = ["price", "count", "add_cost"]
        for col in numeric_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                if fit:
                    # Сохраняем медиану для future use
                    median_val = df_clean[col].median()
                    self._preprocessing_params[f"{col}_median"] = median_val
                else:
                    # Используем сохраненную медиану
                    median_val = self._preprocessing_params.get(
                        f"{col}_median", df_clean[col].median()
                    )

                df_clean[col].fillna(median_val, inplace=True)
                self.logger.info(f"Заполнены пропуски в {col} медианой: {median_val}")

        # Для категориальных колонок
        categorical_cols = ["company", "product"]
        for col in categorical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                if fit:
                    mode_val = (
                        df_clean[col].mode()[0]
                        if not df_clean[col].mode().empty
                        else "Unknown"
                    )
                    self._preprocessing_params[f"{col}_mode"] = mode_val
                else:
                    mode_val = self._preprocessing_params.get(f"{col}_mode", "Unknown")

                df_clean[col].fillna(mode_val, inplace=True)
                self.logger.info(f"Заполнены пропуски в {col} модой: {mode_val}")

        return df_clean

    def _handle_outliers(
        self, df: pd.DataFrame, fit: bool = True, method: str = "iqr"
    ) -> pd.DataFrame:
        """
        Обработка выбросов.
        """
        df_clean = df.copy()
        numeric_cols = ["price", "count", "add_cost"]

        for col in numeric_cols:
            if col not in df_clean.columns:
                continue

            if method == "iqr":
                if fit:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1

                    self._preprocessing_params[f"{col}_Q1"] = Q1
                    self._preprocessing_params[f"{col}_Q3"] = Q3
                    self._preprocessing_params[f"{col}_IQR"] = IQR
                else:
                    Q1 = self._preprocessing_params.get(
                        f"{col}_Q1", df_clean[col].quantile(0.25)
                    )
                    Q3 = self._preprocessing_params.get(
                        f"{col}_Q3", df_clean[col].quantile(0.75)
                    )
                    IQR = self._preprocessing_params.get(f"{col}_IQR", Q3 - Q1)

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                if outliers.sum() > 0:
                    self.logger.info(f"Найдено {outliers.sum()} выбросов в {col}")
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

        return df_clean

    def get_preprocessing_params(self) -> Dict[str, Any]:
        """Получение параметров предобработки."""
        return self._preprocessing_params

    def set_preprocessing_params(self, params: Dict[str, Any]) -> None:
        """Установка параметров предобработки."""
        self._preprocessing_params = params

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Получение сводной статистики по данным.
        """
        if self.data is None:
            return {}

        numeric_cols = ["price", "count", "add_cost"]
        stats = {
            "total_rows": len(self.data),
            "columns": list(self.data.columns),
            "numeric_stats": {},
        }

        for col in numeric_cols:
            if col in self.data.columns:
                stats["numeric_stats"][col] = {
                    "mean": float(self.data[col].mean()),
                    "std": float(self.data[col].std()),
                    "min": float(self.data[col].min()),
                    "max": float(self.data[col].max()),
                    "q25": float(self.data[col].quantile(0.25)),
                    "q50": float(self.data[col].quantile(0.50)),
                    "q75": float(self.data[col].quantile(0.75)),
                }

        stats["categorical_stats"] = {}
        for col in ["company", "product"]:
            if col in self.data.columns:
                stats["categorical_stats"][col] = {
                    "unique_count": self.data[col].nunique(),
                    "top_values": self.data[col].value_counts().head(5).to_dict(),
                }

        return stats
