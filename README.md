# Модель ценообразования

## Описание

Разработка алгоритма прогнозирования цен на товары на основе исторических данных, включающих цены, объемы продаж, затраты на продвижение и информацию о конкурентах.

## Особенности

-  Загрузка и предобработка данных из CSV
-  Хранение данных в SQLite
-  Создание признаков и обучение модели
-  Регулярное обновление модели
-  Оценка качества прогнозирования
-  Соответствие PEP8
-  Покрытие тестами > 75%

## Требования

- Python 3.11+
- Pandas, NumPy, scikit-learn
- SQLite3

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/price-prediction.git
cd price-prediction

# Установка зависимостей
pip install -r requirements.txt

# Или через Poetry
poetry install
Использование
Командная строка
bash
# Обучение модели
python main.py --train --data data/csv_data.csv

# Прогнозирование цены
python main.py --predict --company Amazon --product ThinkPad --count 500 --add_cost 3000

# Обновление модели
python main.py --update --data data/new_data.csv

# Запуск периодических обновлений
python main.py --schedule --data data/new_data.csv
Программный интерфейс
python
from price_prediction import PricePredictor

# Создание предиктора
predictor = PricePredictor()

# Загрузка и обучение
predictor.load_and_prepare_data('data/csv_data.csv')
metrics = predictor.train_model()

# Прогнозирование
price = predictor.predict({
    'company': 'Amazon',
    'product': 'ThinkPad',
    'count': 500,
    'add_cost': 3000
})
print(f'Предсказанная цена: {price}')
Тестирование
bash
# Запуск всех тестов
pytest tests/ -v

# Запуск с покрытием
pytest tests/ --cov=price_prediction --cov-report=term-missing

Структура проекта
text
price-prediction/
 price_prediction/      # Основной код
  data_loader.py     # Загрузка данных
  database_manager.py # Управление БД
  feature_engineer.py # Создание признаков
  price_predictor.py  # Прогнозирование цен
  model_updater.py    # Обновление модели
  utils.py           # Утилиты
 tests/                 # Тесты
 data/                  # Данные
 models/                # Сохраненные модели
 main.py               # Главный скрипт
Метрики качества
•	R² (коэффициент детерминации)
•	MAE (средняя абсолютная ошибка)
•	RMSE (среднеквадратичная ошибка)
•	MAPE (средняя абсолютная процентная ошибка)
Лицензия
MIT
text

Это полный код проекта с хорошей архитектурой, тестами и документацией. Проект готов к использованию и легко расширяется.

Контакты
По вопросам сотрудничества обращайтесь по электронной почте: Pavelru163@gmail.com