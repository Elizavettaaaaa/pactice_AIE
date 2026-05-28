# Конфигурационные файлы проекта

## Описание

Папка `configs/` содержит файлы конфигурации для проекта прогнозирования оттока клиентов.

## Файлы

### config.yaml

Основной конфигурационный файл, содержащий:

- **data** — пути к данным, размер тестовой выборки, random_state
- **categorical_features** — список категориальных признаков для One-Hot Encoding
- **numeric_features** — список числовых признаков для масштабирования
- **models** — параметры моделей (Logistic Regression, Random Forest, XGBoost, CatBoost)
- **grid_search** — сетки гиперпараметров для подбора
- **api** — настройки API сервиса (хост, порт, название, версия)
- **threshold** — настройки порога классификации
- **logging** — настройки логирования
- **docker** — настройки Docker образа

## Использование

В проекте конфигурация загружается через модуль `configs/config.yaml`. 
Пример загрузки в Python:

```python
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

# Доступ к параметрам
data_path = config['data']['raw_path']
model_params = config['models']['xgboost']