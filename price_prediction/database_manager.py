"""Модуль для управления базой данных."""

import sqlite3
import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class DatabaseManager:
    """
    Класс для управления хранением данных в SQLite.
    """

    def __init__(self, db_path: str = 'price_data.db'):
        """
        Инициализация менеджера базы данных.

        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Инициализация базы данных и создание таблиц."""
        with self._get_connection() as conn:
            # Создание основной таблицы
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    price REAL,
                    count INTEGER,
                    add_cost REAL,
                    company TEXT,
                    product TEXT,
                    price_normalized REAL,
                    count_normalized REAL,
                    add_cost_normalized REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Создание таблицы для метаданных модели
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    training_data_count INTEGER,
                    features TEXT,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Создание индексов для ускорения запросов
            conn.execute('CREATE INDEX IF NOT EXISTS idx_company ON price_history(company)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_product ON price_history(product)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON price_history(created_at)')

            self.logger.info(f"База данных инициализирована: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """
        Контекстный менеджер для подключения к БД.

        Yields:
            sqlite3.Connection: Подключение к БД
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Ошибка при работе с БД: {e}")
            raise
        finally:
            conn.close()

    def insert_batch(self, df: pd.DataFrame) -> int:
        """
        Вставка данных пачками.

        Args:
            df: DataFrame с данными для вставки

        Returns:
            int: Количество вставленных записей
        """
        if df.empty:
            self.logger.warning("Пустой DataFrame для вставки")
            return 0

        # Обязательные колонки (всегда должны быть)
        required_columns = ['price', 'count', 'add_cost']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Определяем, какие колонки из необязательных есть в DataFrame
        optional_columns = ['company', 'product']
        available_optional = [col for col in optional_columns if col in df.columns]

        # Все колонки для вставки
        insert_columns = required_columns + available_optional

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Подготовка данных
            records = df[insert_columns].to_dict('records')

            # Создаем SQL запрос динамически
            placeholders = ', '.join(['?' for _ in insert_columns])
            columns_str = ', '.join(insert_columns)

            # Вставка записей
            inserted = 0
            for record in records:
                try:
                    query = f'''
                        INSERT INTO price_history 
                        ({columns_str})
                        VALUES ({placeholders})
                    '''
                    # Формируем значения в правильном порядке
                    values = tuple(record[col] for col in insert_columns)
                    cursor.execute(query, values)
                    inserted += 1
                except Exception as e:
                    self.logger.error(f"Ошибка при вставке записи: {e}")
                    continue

            self.logger.info(f"Вставлено {inserted} записей в базу данных")
            return inserted

    def get_all_data(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            company: Optional[str] = None,
            product: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Получение всех данных из базы.

        Args:
            limit: Лимит записей
            offset: Смещение
            company: Фильтр по компании
            product: Фильтр по продукту

        Returns:
            pd.DataFrame: Данные из базы
        """
        query = '''
            SELECT price, count, add_cost, company, product, created_at
            FROM price_history
            WHERE 1=1
        '''
        params = []

        if company:
            query += ' AND company = ?'
            params.append(company)

        if product:
            query += ' AND product = ?'
            params.append(product)

        query += ' ORDER BY created_at DESC'

        if limit is not None:
            query += ' LIMIT ? OFFSET ?'
            params.extend([limit, offset])

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            self.logger.info(f"Получено {len(df)} записей из базы данных")
            return df

    def get_latest_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Получение последних данных.

        Args:
            limit: Количество последних записей

        Returns:
            pd.DataFrame: Последние данные
        """
        return self.get_all_data(limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по данным в БД.

        Returns:
            Dict[str, Any]: Статистика
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Общее количество записей
            cursor.execute('SELECT COUNT(*) FROM price_history')
            total_count = cursor.fetchone()[0]

            # Статистика по компаниям
            cursor.execute('''
                SELECT company, COUNT(*) as count, AVG(price) as avg_price
                FROM price_history
                GROUP BY company
                ORDER BY count DESC
            ''')
            company_stats = [dict(row) for row in cursor.fetchall()]

            # Статистика по продуктам
            cursor.execute('''
                SELECT product, COUNT(*) as count, AVG(price) as avg_price
                FROM price_history
                GROUP BY product
                ORDER BY count DESC
                LIMIT 10
            ''')
            product_stats = [dict(row) for row in cursor.fetchall()]

            return {
                'total_records': total_count,
                'company_stats': company_stats,
                'product_stats': product_stats
            }

    def clear_data(self) -> None:
        """Очистка всех данных из таблицы."""
        with self._get_connection() as conn:
            conn.execute('DELETE FROM price_history')
            self.logger.info("Все данные очищены из базы данных")

    def save_model_metadata(
            self,
            model_version: str,
            training_count: int,
            features: List[str],
            metrics: Dict[str, float]
    ) -> int:
        """
        Сохранение метаданных модели.

        Args:
            model_version: Версия модели
            training_count: Количество записей для обучения
            features: Список признаков
            metrics: Метрики модели

        Returns:
            int: ID сохраненной записи
        """
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_metadata
                (model_version, training_data_count, features, metrics)
                VALUES (?, ?, ?, ?)
            ''', (
                model_version,
                training_count,
                json.dumps(features),
                json.dumps(metrics)
            ))
            return cursor.lastrowid
