"""Основной модуль для прогнозирования цен."""

import pandas as pd
import logging
import pickle
import os
from typing import Optional, Dict, Any, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from price_prediction.data_loader import DataLoader
from price_prediction.database_manager import DatabaseManager
from price_prediction.feature_engineer import FeatureEngineer
from price_prediction.utils import calculate_metrics, save_model_metadata
import numpy as np
from sklearn.impute import SimpleImputer


class PricePredictor:
    """
    Класс для прогнозирования цен на основе исторических данных.
    """

    def __init__(
            self,
            model_type: str = 'ridge',
            db_path: str = 'price_data.db',
            model_dir: str = 'models'
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
            random_state: int = 42
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

        # Подготовка данных для модели
        X, y = self._prepare_training_data(data)

        nan_count_X = X.isnull().sum().sum()
        if nan_count_X > 0:
            self.logger.warning(f"Обнаружено {nan_count_X} NaN значений в признаках. Заполняем...")

            # Заполняем NaN средним значением каждой колонки
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            X = X_imputed

            # Проверяем, что NaN нет
            if X.isnull().any().any():
                self.logger.warning("Остались NaN после заполнения! Заполняем нулями.")
                X = X.fillna(0)

        # Проверка на бесконечные значения
        if np.isinf(X.values).any():
            self.logger.warning("Обнаружены бесконечные значения. Заменяем на 0.")
            X = X.replace([np.inf, -np.inf], 0)

        # Проверка на NaN в целевой переменной
        if np.isnan(y).any():
            nan_count_y = np.isnan(y).sum()
            self.logger.warning(f"Обнаружено {nan_count_y} NaN значений в целевой переменной. Удаляем...")
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]

        # Логируем итоговый размер данных
        self.logger.info(f"Размер данных после очистки: {len(X)} записей")

        # Проверка, что данные не пустые
        if X.shape[0] == 0:
            raise ValueError("Все данные содержат NaN. Проверьте предобработку.")

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Обучение модели
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # Сохранение признаков
        self.feature_columns = X.columns.tolist()

        # Оценка модели
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Детальные метрики
        y_pred = self.model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        metrics['train_r2'] = float(train_score)
        metrics['test_r2'] = float(test_score)

        self.logger.info(f"Обучение завершено. R2 на тесте: {test_score:.4f}")

        # Сохранение модели
        self._save_model(metrics)

        return metrics

    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Подготовка данных для обучения.

        Args:
            data: Исходные данные

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Признаки и целевая переменная
        """
        # Создание признаков
        X, feature_cols = self.feature_engineer.prepare_for_model(data, fit=True)

        # Целевая переменная
        y = data['price'].values

        self.logger.info(f"Подготовлено {len(X)} записей, {len(feature_cols)} признаков")
        return X, y

    def _create_model(self):
        """
        Создание модели машинного обучения.

        Returns:
            sklearn.base.BaseEstimator: Модель
        """
        if self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.model_type}")

    def predict(
            self,
            features: Dict[str, Any],
            use_latest_data: bool = False
    ) -> float:
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
            self,
            features: Dict[str, Any],
            use_latest_data: bool = False
    ) -> pd.DataFrame:
        """
        Подготовка данных для прогноза.

        Args:
            features: Словарь с признаками
            use_latest_data: Использовать последние данные для контекста

        Returns:
            pd.DataFrame: Подготовленные данные
        """
        # Создание DataFrame из словаря
        df = pd.DataFrame([features])

        hist_data = self.db_manager.get_all_data()
        if not hist_data.empty:
            context_features = self._create_context_features(hist_data)
            for col, value in context_features.items():
                df[col] = value
        else:
            self.logger.warning("Нет исторических данных для создания контекстных признаков")

        # Подготовка данных
        X, _ = self.feature_engineer.prepare_for_model(df, fit=False)

        #  Заполнение недостающих признаков нулями
        for col in self.feature_columns:
            if col not in X.columns:
                self.logger.warning(f"Признак {col} отсутствует, заполняем нулем")
                X[col] = 0

        # Оставляем только нужные признаки в правильном порядке
        X = X[self.feature_columns]

        return X

    def _create_context_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Создание контекстных признаков из исторических данных.
        Создаем ВСЕ признаки, которые использует модель.
        """
        import numpy as np
        from datetime import datetime

        context = {}

        # ОСНОВНЫЕ ПРИЗНАКИ
        # Используем средние значения из исторических данных
        if 'count' in df.columns:
            context['count'] = df['count'].mean()
        if 'add_cost' in df.columns:
            context['add_cost'] = df['add_cost'].mean()

        # ЛОГАРИФМИЧЕСКИЕ ПРИЗНАКИ
        epsilon = 1e-10

        if 'price' in df.columns:
            avg_price = df['price'].mean()
            if avg_price <= 0:
                shift = abs(avg_price) + 1
                context['log_price'] = np.log1p(avg_price + shift)
            else:
                context['log_price'] = np.log1p(avg_price)

        if 'count' in df.columns:
            avg_count = df['count'].mean()
            if avg_count <= 0:
                shift = abs(avg_count) + 1
                context['log_count'] = np.log1p(avg_count + shift)
            else:
                context['log_count'] = np.log1p(avg_count)

        if 'add_cost' in df.columns:
            avg_add_cost = df['add_cost'].mean()
            if avg_add_cost <= 0:
                shift = abs(avg_add_cost) + 1
                context['log_add_cost'] = np.log1p(avg_add_cost + shift)
            else:
                context['log_add_cost'] = np.log1p(avg_add_cost)

        # ОТНОШЕНИЯ ПРИЗНАКОВ
        if 'price' in df.columns and 'count' in df.columns:
            avg_price = df['price'].mean()
            avg_count = df['count'].mean()
            context['price_per_count'] = avg_price / (avg_count + epsilon)

        if 'price' in df.columns and 'add_cost' in df.columns:
            avg_price = df['price'].mean()
            avg_add_cost = df['add_cost'].mean()
            context['price_per_cost'] = avg_price / (avg_add_cost + epsilon)

        if 'count' in df.columns and 'add_cost' in df.columns:
            avg_count = df['count'].mean()
            avg_add_cost = df['add_cost'].mean()
            context['cost_per_count'] = avg_add_cost / (avg_count + epsilon)

        # ПРИЗНАКИ ВЗАИМОДЕЙСТВИЯ
        if 'price' in df.columns and 'count' in df.columns:
            avg_price = df['price'].mean()
            avg_count = df['count'].mean()
            context['price_x_count'] = avg_price * avg_count

        if 'price' in df.columns and 'add_cost' in df.columns:
            avg_price = df['price'].mean()
            avg_add_cost = df['add_cost'].mean()
            context['price_x_add_cost'] = avg_price * avg_add_cost

        if 'count' in df.columns and 'add_cost' in df.columns:
            avg_count = df['count'].mean()
            avg_add_cost = df['add_cost'].mean()
            context['count_x_add_cost'] = avg_count * avg_add_cost

        # АГРЕГИРОВАННЫЕ ПРИЗНАКИ (company и product)
        if 'price' in df.columns:
            context['company_price_mean'] = df['price'].mean()
            context['company_price_median'] = df['price'].median()
            context['company_price_std'] = df['price'].std()
            context['company_price_min'] = df['price'].min()
            context['company_price_max'] = df['price'].max()

            context['product_price_mean'] = df['price'].mean()
            context['product_price_median'] = df['price'].median()
            context['product_price_std'] = df['price'].std()
            context['product_price_min'] = df['price'].min()
            context['product_price_max'] = df['price'].max()

        # ВРЕМЕННЫЕ ПРИЗНАКИ
        now = datetime.now()
        context['year'] = now.year
        context['month'] = now.month
        context['day'] = now.day
        context['dayofweek'] = now.weekday()
        context['quarter'] = (now.month - 1) // 3 + 1
        context['month_sin'] = np.sin(2 * np.pi * now.month / 12)
        context['month_cos'] = np.cos(2 * np.pi * now.month / 12)
        context['dayofweek_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
        context['dayofweek_cos'] = np.cos(2 * np.pi * now.weekday() / 7)

        return context

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.

        Args:
            test_data: Тестовые данные

        Returns:
            Dict[str, float]: Метрики качества
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        X, y = self._prepare_training_data(test_data)
        y_pred = self.model.predict(X)

        metrics = calculate_metrics(y, y_pred)

        self.logger.info(f"Оценка модели на тестовых данных: R2={metrics['r2']:.4f}")
        return metrics

    def _save_model(self, metrics: Dict[str, float]) -> None:
        """
        Сохранение модели на диск.

        Args:
            metrics: Метрики модели
        """
        if self.model is None:
            return

        model_path = os.path.join(self.model_dir, 'model.pkl')
        feature_path = os.path.join(self.model_dir, 'features.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')

        # Сохранение модели
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns
            }, f)

        # Сохранение признаков
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)

        #  СОХРАНЕНИЕ СКАЛЕРА
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_engineer.scaler, f)
            self.logger.info(f"Скалер сохранен в {scaler_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении скалера: {e}")

        #  СОХРАНЕНИЕ LABEL ENCODERS
        try:
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.feature_engineer.label_encoders, f)
            self.logger.info(f"LabelEncoders сохранены в {encoders_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении энкодеров: {e}")

        # Сохранение метаданных в БД
        self.db_manager.save_model_metadata(
            model_version=self.model_type,
            training_count=len(self.feature_columns),
            features=self.feature_columns,
            metrics=metrics
        )

        # Сохранение метаданных в файл
        save_model_metadata(model_path, metrics)

        self.logger.info(f"Модель сохранена в {model_path}")

    def _load_model(self) -> bool:
        """
        Загрузка сохраненной модели.

        Returns:
            bool: True если загрузка успешна
        """
        model_path = os.path.join(self.model_dir, 'model.pkl')
        feature_path = os.path.join(self.model_dir, 'features.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')

        #  СНАЧАЛА проверяем, что модель существует
        if not os.path.exists(model_path):
            self.logger.warning("Сохраненная модель не найдена")
            return False

        try:
            # Загрузка модели
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.model_type = data.get('model_type', self.model_type)
                self.feature_columns = data.get('feature_columns', [])

            # Загрузка признаков
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)

            #  Загрузка скалера (если есть)
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_engineer.scaler = pickle.load(f)
                    self.logger.info("Скалер загружен")

            #  Загрузка LabelEncoders (если есть метод)
            if hasattr(self.feature_engineer, 'load_encoders'):
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
            'model_type': self.model_type,
            'is_trained': self.model is not None,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns
        }

        if self.model is not None:
            if hasattr(self.model, 'coef_'):
                info['coefficients'] = dict(zip(self.feature_columns, self.model.coef_))
            if hasattr(self.model, 'feature_importances_'):
                info['feature_importances'] = dict(
                    zip(self.feature_columns, self.model.feature_importances_)
                )

        return info
