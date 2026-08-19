"""Основной модуль для прогнозирования цен."""

import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from price_prediction.data_loader import DataLoader
from price_prediction.database_manager import DatabaseManager
from price_prediction.feature_engineer import FeatureEngineer
from price_prediction.utils import calculate_metrics, save_model_metadata


class PricePredictor:
    """
    Класс для прогнозирования цен на основе исторических данных.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        db_path: str = "price_data.db",
        model_dir: str = "models",
    ):
        """
        Инициализация предиктора цен.

        Args:
            model_type: Тип модели ('ridge' или 'random_forest')
            db_path: Путь к базе данных
            model_dir: Директория для сохранения моделей
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.db_manager = DatabaseManager(db_path)
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.feature_columns = []
        self.scaler = None
        self.imputer = None
        self.logger = logging.getLogger(__name__)

        # Создание директории для моделей
        os.makedirs(model_dir, exist_ok=True)

        self.logger.info(f"Инициализация PricePredictor с моделью {model_type}")

    def load_and_prepare_data(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка и подготовка данных.

        Args:
            filepath: Путь к CSV файлу

        Returns:
            pd.DataFrame: Подготовленные данные
        """
        # Загрузка данных
        loader = DataLoader(filepath)
        df = loader.load_data()

        self.db_manager.clear_data()  # Удаляем старые данные

        self.db_manager.insert_batch(df)

        # Предобработка (для использования в текущей сессии)
        df_processed = loader.preprocess_data(df)

        self.logger.info(f"Загружено и обработано {len(df_processed)} записей")
        return df_processed

    def train_model(
        self,
        data: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Обучение модели прогнозирования.

        Args:
            data: Данные для обучения (если None, загружаются из БД)
            test_size: Размер тестовой выборки
            random_state: Seed для воспроизводимости

        Returns:
            Dict[str, float]: Метрики качества модели
        """
        if data is None:
            # Загрузка данных из БД
            data = self.db_manager.get_all_data()

        if data.empty:
            raise ValueError("Нет данных для обучения модели")

        self.logger.info(f"Начало обучения модели. Размер данных: {len(data)}")

        # Разделение на train и test ДО любого преобразования
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        self.logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

        # Подготовка данных для обучения (fit=True для обучения трансформеров)
        X_train, y_train = self._prepare_training_data(train_data, fit=True)

        # Подготовка тестовых данных (fit=False для применения уже обученных трансформеров)
        X_test, y_test = self._prepare_training_data(test_data, fit=False)

        # Проверка и обработка пропусков в train
        X_train, y_train = self._clean_data(X_train, y_train, fit=True)

        # Проверка и обработка пропусков в test
        X_test, y_test = self._clean_data(X_test, y_test, fit=False)

        # Обучение модели
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Сохранение признаков
        self.feature_columns = X_train.columns.tolist()

        # Оценка модели
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Детальные метрики на тестовых данных
        y_pred = self.model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        metrics["train_r2"] = float(train_score)
        metrics["test_r2"] = float(test_score)

        self.logger.info(f"Обучение завершено. R2 на тесте: {test_score:.4f}")

        # Сохранение модели и трансформеров
        self._save_model(metrics)

        return metrics

    def _clean_data(
        self, X: pd.DataFrame, y: np.ndarray, fit: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Очистка данных от NaN и бесконечных значений.

        Args:
            X: Признаки
            y: Целевая переменная
            fit: True для обучения импьютера

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Очищенные данные
        """
        # Проверка на NaN в признаках
        if X.isnull().any().any():
            self.logger.warning("Обнаружены NaN значения в признаках. Заполняем...")

            if fit:
                self.imputer = SimpleImputer(strategy="mean")
                X_imputed = pd.DataFrame(
                    self.imputer.fit_transform(X), columns=X.columns, index=X.index
                )
            else:
                if self.imputer is None:
                    raise ValueError("Imputer не обучен. Сначала обучите модель.")
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X), columns=X.columns, index=X.index
                )
            X = X_imputed

        # Проверка на бесконечные значения
        if np.isinf(X.values).any():
            self.logger.warning("Обнаружены бесконечные значения. Заменяем на 0.")
            X = X.replace([np.inf, -np.inf], 0)

        # Проверка на NaN в целевой переменной
        if np.isnan(y).any():
            nan_count_y = np.isnan(y).sum()
            self.logger.warning(
                f"Обнаружено {nan_count_y} NaN значений в целевой переменной. Удаляем..."
            )
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]

        if X.shape[0] == 0:
            raise ValueError("Все данные содержат NaN. Проверьте предобработку.")

        return X, y

    def _prepare_training_data(
        self, data: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Подготовка данных для обучения.

        Args:
            data: Исходные данные
            fit: True для обучения трансформеров, False для применения существующих

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Признаки и целевая переменная
        """
        # ВАЖНО: Сохраняем целевую переменную ДО удаления
        y = data["price"].values

        # Удаляем price из данных перед созданием признаков
        X_data = data.drop(columns=["price"], errors="ignore")

        # Создание признаков (без использования price)
        X = self.feature_engineer.create_features(X_data, fit=fit)

        # Кодирование категориальных переменных
        X = self.feature_engineer.encode_categorical(X, fit=fit)

        # Масштабирование признаков
        X = self.feature_engineer.scale_features(X, fit=fit)

        # Выбор признаков для модели
        feature_cols = self.feature_engineer._select_model_features(X)
        X = X[feature_cols]

        try:
            # Создаем временный DataFrame с признаками и ценой
            temp_df = X.copy()
            temp_df["price"] = y

            # Вычисляем корреляции
            correlations = (
                temp_df.corr()["price"].drop("price").sort_values(ascending=False)
            )
            self.logger.info("=== ТОП-10 КОРРЕЛЯЦИЙ С ЦЕНОЙ ===")
            for feat, corr in correlations.head(10).items():
                self.logger.info(f"  {feat}: {corr:.4f}")
            self.logger.info("==================================")

            if correlations.abs().max() < 0.1:
                self.logger.warning(
                    "Максимальная корреляция < 0.1! Признаки не связаны с ценой."
                )
        except Exception as e:
            self.logger.warning(f"Не удалось вычислить корреляции: {e}")

        self.logger.info(
            f"Подготовлено {len(X)} записей, {len(feature_cols)} признаков"
        )
        return X, y

    def _create_model(self):
        """
        Создание модели машинного обучения.

        Returns:
            sklearn.base.BaseEstimator: Модель
        """
        if self.model_type == "random_forest":
            return Ridge(alpha=100.0, random_state=42)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")

    def predict(self, features: Dict[str, Any], use_latest_data: bool = False) -> float:
        """
        Прогнозирование цены.

        Args:
            features: Словарь с признаками
            use_latest_data: Использовать последние данные для контекста

        Returns:
            float: Предсказанная цена
        """
        if self.model is None:
            # Попытка загрузить сохраненную модель
            if not self._load_model():
                raise ValueError("Модель не обучена")

        # Подготовка данных для прогноза
        feature_df = self._prepare_prediction_data(features, use_latest_data)

        # Прогнозирование
        prediction = self.model.predict(feature_df)[0]

        self.logger.info(f"Прогноз цены: {prediction:.2f}")
        return float(prediction)

    def _prepare_prediction_data(
        self, features: Dict[str, Any], use_latest_data: bool = False
    ) -> pd.DataFrame:
        df = pd.DataFrame([features])

        # Если use_latest_data=True - берём только последние данные
        if use_latest_data:
            hist_data = self.db_manager.get_latest_data(limit=1000)  # Или другой лимит
            self.logger.info(
                f"Использованы последние {len(hist_data)} записей для контекста"
            )
        else:
            hist_data = self.db_manager.get_all_data()
            self.logger.info(f"Использованы все {len(hist_data)} записей для контекста")

        if not hist_data.empty:
            context_features = self._create_context_features(hist_data)
            for col, value in context_features.items():
                df[col] = value
        else:
            self.logger.warning(
                "Нет исторических данных для создания контекстных признаков"
            )

        # Подготовка данных (fit=False - используем существующие трансформеры)
        X = self.feature_engineer.prepare_for_prediction(df, fit=False)

        # Проверка наличия всех признаков
        for col in self.feature_columns:
            if col not in X.columns:
                self.logger.warning(f"Признак {col} отсутствует, заполняем нулем")
                X[col] = 0

        # Оставляем только нужные признаки в правильном порядке
        X = X[self.feature_columns]

        return X

    def _create_context_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Создание контекстных признаков из исторических данных."""
        from datetime import datetime

        import numpy as np

        context = {}

        # ТОЛЬКО ТЕ ПРИЗНАКИ, КОТОРЫЕ ИСПОЛЬЗУЕТ МОДЕЛЬ
        if "count" in df.columns:
            context["count"] = df["count"].mean()
        if "add_cost" in df.columns:
            context["add_cost"] = df["add_cost"].mean()

        # Логарифмы
        epsilon = 1e-10
        if "count" in df.columns:
            avg_count = df["count"].mean()
            if avg_count <= 0:
                shift = abs(avg_count) + 1
                context["log_count"] = np.log1p(avg_count + shift)
            else:
                context["log_count"] = np.log1p(avg_count)

        if "add_cost" in df.columns:
            avg_add_cost = df["add_cost"].mean()
            if avg_add_cost <= 0:
                shift = abs(avg_add_cost) + 1
                context["log_add_cost"] = np.log1p(avg_add_cost + shift)
            else:
                context["log_add_cost"] = np.log1p(avg_add_cost)

        # Взаимодействия
        if "count" in df.columns and "add_cost" in df.columns:
            avg_count = df["count"].mean()
            avg_add_cost = df["add_cost"].mean()
            context["cost_per_count"] = avg_add_cost / (avg_count + epsilon)
            context["count_x_add_cost"] = avg_count * avg_add_cost

        # Временные признаки (из текущей даты)
        now = datetime.now()
        context["year"] = now.year
        context["month"] = now.month
        context["day"] = now.day
        context["dayofweek"] = now.weekday()
        context["quarter"] = (now.month - 1) // 3 + 1
        context["month_sin"] = np.sin(2 * np.pi * now.month / 12)
        context["month_cos"] = np.cos(2 * np.pi * now.month / 12)
        context["dayofweek_sin"] = np.sin(2 * np.pi * now.weekday() / 7)
        context["dayofweek_cos"] = np.cos(2 * np.pi * now.weekday() / 7)

        return context

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.
        Использует уже обученные трансформеры (fit=False).

        Args:
            test_data: Тестовые данные

        Returns:
            Dict[str, float]: Метрики качества
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        # Используем fit=False для применения существующих трансформеров
        X, y = self._prepare_training_data(test_data, fit=False)

        # Очистка данных с использованием существующего импьютера
        X, y = self._clean_data(X, y, fit=False)

        y_pred = self.model.predict(X)

        metrics = calculate_metrics(y, y_pred)

        self.logger.info(f"Оценка модели на тестовых данных: R2={metrics['r2']:.4f}")
        return metrics

    def _save_model(self, metrics: Dict[str, float]) -> None:
        """
        Сохранение модели и всех трансформеров на диск.

        Args:
            metrics: Метрики модели
        """
        if self.model is None:
            return

        model_path = os.path.join(self.model_dir, "model.pkl")
        feature_path = os.path.join(self.model_dir, "features.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        imputer_path = os.path.join(self.model_dir, "imputer.pkl")

        # Сохранение модели
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "model_type": self.model_type,
                    "feature_columns": self.feature_columns,
                },
                f,
            )

        # Сохранение признаков
        with open(feature_path, "wb") as f:
            pickle.dump(self.feature_columns, f)

        # Сохранение скалера
        if self.feature_engineer.scaler is not None:
            with open(scaler_path, "wb") as f:
                pickle.dump(self.feature_engineer.scaler, f)
            self.logger.info(f"Скалер сохранен в {scaler_path}")

        # Сохранение импьютера
        if self.imputer is not None:
            with open(imputer_path, "wb") as f:
                pickle.dump(self.imputer, f)
            self.logger.info(f"Imputer сохранен в {imputer_path}")

        # Сохранение LabelEncoders
        self.feature_engineer.save_encoders(self.model_dir)

        # Сохранение метаданных в БД
        self.db_manager.save_model_metadata(
            model_version=self.model_type,
            training_count=len(self.feature_columns),
            features=self.feature_columns,
            metrics=metrics,
        )

        # Сохранение метаданных в файл
        save_model_metadata(model_path, metrics)

        self.logger.info(f"Модель сохранена в {model_path}")

    def _load_model(self) -> bool:
        """
        Загрузка сохраненной модели и всех трансформеров.

        Returns:
            bool: True если загрузка успешна
        """
        model_path = os.path.join(self.model_dir, "model.pkl")
        feature_path = os.path.join(self.model_dir, "features.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        imputer_path = os.path.join(self.model_dir, "imputer.pkl")

        if not os.path.exists(model_path):
            self.logger.warning("Сохраненная модель не найдена")
            return False

        try:
            # Загрузка модели
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.model_type = data.get("model_type", self.model_type)
                self.feature_columns = data.get("feature_columns", [])

            # Загрузка признаков
            if os.path.exists(feature_path):
                with open(feature_path, "rb") as f:
                    self.feature_columns = pickle.load(f)

            # Загрузка скалера
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.feature_engineer.scaler = pickle.load(f)
                    self.logger.info("Скалер загружен")

            # Загрузка импьютера
            if os.path.exists(imputer_path):
                with open(imputer_path, "rb") as f:
                    self.imputer = pickle.load(f)
                    self.logger.info("Imputer загружен")

            # Загрузка LabelEncoders
            self.feature_engineer.load_encoders(self.model_dir)

            self.logger.info(f"Модель загружена из {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели.

        Returns:
            Dict[str, Any]: Информация о модели
        """
        info = {
            "model_type": self.model_type,
            "is_trained": self.model is not None,
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
        }

        if self.model is not None:
            if hasattr(self.model, "coef_"):
                info["coefficients"] = dict(zip(self.feature_columns, self.model.coef_))
            if hasattr(self.model, "feature_importances_"):
                info["feature_importances"] = dict(
                    zip(self.feature_columns, self.model.feature_importances_)
                )

        return info
