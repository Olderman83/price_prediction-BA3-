"""Модуль для регулярного обновления модели."""

import time
import logging
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import schedule
import threading
from price_prediction.price_predictor import PricePredictor
from price_prediction.data_loader import DataLoader


class ModelUpdater:
    """
    Класс для регулярного обновления модели с новыми данными.
    """

    def __init__(
            self,
            predictor: Optional[PricePredictor] = None,
            data_path: Optional[str] = None,
            update_interval: int = 3600  # в секундах
    ):
        """
        Инициализация обновлятора модели.

        Args:
            predictor: Экземпляр PricePredictor
            data_path: Путь к данным для обновления
            update_interval: Интервал обновления в секундах
        """
        self.predictor = predictor or PricePredictor()
        self.data_path = data_path
        self.update_interval = update_interval
        self.is_running = False
        self.last_update = None
        self.logger = logging.getLogger(__name__)

    def update_model(self, data_path: Optional[str] = None) -> bool:
        """
        Обновление модели новыми данными.

        Args:
            data_path: Путь к новым данным

        Returns:
            bool: True если обновление успешно
        """
        try:
            self.logger.info("Начало обновления модели")

            # Загрузка новых данных
            if data_path:
                self.data_path = data_path

            if not self.data_path:
                self.logger.warning("Нет пути к данным для обновления")
                return False

            # Загрузка и подготовка данных
            loader = DataLoader(self.data_path)
            new_data = loader.load_data()
            new_data_processed = loader.preprocess_data(new_data)

            # Получение существующих данных из БД
            existing_data = self.predictor.db_manager.get_all_data()

            # Объединение данных
            if not existing_data.empty:
                combined_data = pd.concat([existing_data, new_data_processed])
                # Удаление дубликатов
                combined_data = combined_data.drop_duplicates(
                    subset=['company', 'product', 'price', 'count', 'add_cost']
                )
            else:
                combined_data = new_data_processed

            # Сохранение новых данных в БД
            self.predictor.db_manager.insert_batch(new_data_processed)

            # Переобучение модели
            metrics = self.predictor.train_model(combined_data)

            self.last_update = datetime.now()
            self.logger.info(f"Модель обновлена успешно. Метрики: {metrics}")

            return True

        except Exception as e:
            self.logger.error(f"Ошибка при обновлении модели: {e}")
            return False

    def schedule_updates(self, interval_seconds: Optional[int] = None) -> None:
        """
        Планирование регулярных обновлений.

        Args:
            interval_seconds: Интервал обновления в секундах
        """
        if interval_seconds:
            self.update_interval = interval_seconds

        # Планирование задачи
        schedule.every(self.update_interval).seconds.do(self.update_model)
        self.logger.info(f"Запланировано обновление модели каждые {self.update_interval} секунд")

    def run_periodic_updates(self, run_immediately: bool = True) -> None:
        """
        Запуск периодических обновлений в отдельном потоке.

        Args:
            run_immediately: Запустить обновление сразу
        """
        if self.is_running:
            self.logger.warning("Процесс обновления уже запущен")
            return

        self.is_running = True

        if run_immediately:
            self.update_model()

        def update_loop():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Проверка каждую минуту

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()

        self.logger.info("Запущен процесс периодических обновлений")

    def stop_periodic_updates(self) -> None:
        """Остановка периодических обновлений."""
        self.is_running = False
        self.logger.info("Процесс периодических обновлений остановлен")

    def get_update_status(self) -> Dict[str, Any]:
        """
        Получение статуса обновлений.

        Returns:
            Dict[str, Any]: Информация о статусе
        """
        status = {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_interval': self.update_interval,
            'data_path': self.data_path
        }

        if self.last_update:
            time_since_update = datetime.now() - self.last_update
            status['seconds_since_update'] = time_since_update.total_seconds()
            status['next_update_in'] = max(0, self.update_interval - time_since_update.total_seconds())

        return status

    def manual_update(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ручное обновление модели.

        Args:
            data_path: Путь к данным

        Returns:
            Dict[str, Any]: Результат обновления
        """
        start_time = time.time()
        success = self.update_model(data_path)
        elapsed_time = time.time() - start_time

        return {
            'success': success,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
