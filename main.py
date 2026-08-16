"""Главный скрипт для запуска приложения."""

import argparse
import logging
import sys
from pathlib import Path

from price_prediction import PricePredictor
from price_prediction.model_updater import ModelUpdater
from price_prediction.utils import setup_logging


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description='Модель ценообразования')
    parser.add_argument('--train', action='store_true', help='Обучение модели')
    parser.add_argument('--data', type=str, help='Путь к файлу данных')
    parser.add_argument('--predict', action='store_true', help='Прогнозирование цены')
    parser.add_argument('--company', type=str, help='Название компании')
    parser.add_argument('--product', type=str, help='Название продукта')
    parser.add_argument('--count', type=int, help='Количество продаж')
    parser.add_argument('--add_cost', type=float, help='Затраты на продвижение')
    parser.add_argument('--update', action='store_true', help='Обновление модели')
    parser.add_argument('--schedule', action='store_true', help='Запуск периодических обновлений')
    parser.add_argument('--db_path', type=str, default='price_data.db', help='Путь к БД')
    parser.add_argument('--log_level', type=str, default='INFO', help='Уровень логирования')

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    # Инициализация предиктора
    predictor = PricePredictor(db_path=args.db_path)

    if args.train:
        if not args.data:
            logger.error("Для обучения необходимо указать файл данных")
            sys.exit(1)

        logger.info(f"Обучение модели на данных из {args.data}")
        metrics = predictor.load_and_prepare_data(args.data)
        metrics = predictor.train_model()
        logger.info(f"Обучение завершено. Метрики: {metrics}")

    elif args.predict:
        if not args.company or not args.product:
            logger.error("Для прогноза необходимо указать компанию и продукт")
            sys.exit(1)

        features = {
            'company': args.company,
            'product': args.product,
            'count': args.count or 500,
            'add_cost': args.add_cost or 3000
        }

        try:
            price = predictor.predict(features)
            logger.info(f"Предсказанная цена: {price:.2f}")
            print(f"Предсказанная цена: {price:.2f}")
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании: {e}")
            sys.exit(1)

    elif args.update:
        updater = ModelUpdater(predictor, args.data)
        result = updater.manual_update()
        logger.info(f"Результат обновления: {result}")

    elif args.schedule:
        updater = ModelUpdater(predictor, args.data)
        updater.schedule_updates(3600)  # Обновление каждый час
        updater.run_periodic_updates()
        logger.info("Запущены периодические обновления. Нажмите Ctrl+C для остановки.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            updater.stop_periodic_updates()
            logger.info("Периодические обновления остановлены")

    else:
        # Интерактивный режим
        logger.info("Запуск в интерактивном режиме")
        print("\n=== Модель ценообразования ===\n")
        print("1. Обучение модели")
        print("2. Прогнозирование цены")
        print("3. Обновление модели")
        print("4. Информация о модели")
        print("5. Выход")

        while True:
            choice = input("\nВыберите действие (1-5): ").strip()

            if choice == '1':
                data_path = input("Введите путь к CSV файлу: ")
                try:
                    predictor.load_and_prepare_data(data_path)
                    metrics = predictor.train_model()
                    print(f"Обучение завершено. Метрики: {metrics}")
                except Exception as e:
                    print(f"Ошибка: {e}")

            elif choice == '2':
                company = input("Введите название компании: ")
                product = input("Введите название продукта: ")
                count = int(input("Введите количество продаж: ") or 500)
                add_cost = float(input("Введите затраты на продвижение: ") or 3000)

                try:
                    price = predictor.predict({
                        'company': company,
                        'product': product,
                        'count': count,
                        'add_cost': add_cost
                    })
                    print(f"Предсказанная цена: {price:.2f}")
                except Exception as e:
                    print(f"Ошибка: {e}")

            elif choice == '3':
                data_path = input("Введите путь к новым данным (или нажмите Enter для пропуска): ")
                updater = ModelUpdater(predictor, data_path if data_path else None)
                result = updater.manual_update()
                print(f"Результат обновления: {result}")

            elif choice == '4':
                info = predictor.get_model_info()
                print("\nИнформация о модели:")
                for key, value in info.items():
                    print(f"  {key}: {value}")

            elif choice == '5':
                print("Выход...")
                break

            else:
                print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()
