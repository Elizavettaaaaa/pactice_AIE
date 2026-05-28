# Customer Churn Prediction Service

## Описание проекта

Сервис для прогнозирования оттока клиентов телекоммуникационной компании.
На основе исторических данных о клиенте сервис предсказывает, уйдёт ли клиент в ближайшее время.

Финальная модель: XGBoost (tuned)
Метрики: F1-score = 0.635, Recall = 80.5%


## Требования к системе

- Python 3.12 или выше
- Docker (опционально, для контейнеризации)
- 4 GB RAM
- 2 GB свободного дискового пространства


## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <https://github.com/Elizavettaaaaa/pactice_AIE>
cd pactice_AIE/project
```
### 2. Создание виртуального окружения

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Запуск API сервиса

```bash
uvicorn src.service.app:app --reload --host 0.0.0.0 --port 8000
```

После запуска сервис будет доступен по адресу: http://localhost:8000

## Запуск через Docker

### 1. Сборка Docker образа

```bash
cd project
docker build -f docker/Dockerfile -t churn-prediction-api .
```

### 2. Запуск контейнера

```bash
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
```

### 3. Остановка контейнера

```bash
docker stop churn-api
docker rm churn-api
```

## Проверка работоспособности

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:

```json
{"status":"healthy","model_loaded":true,"timestamp":"...","version":"1.0.0"}
```

## Документация API

После запуска сервиса документация доступна по адресам:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


## Эндпоинты API

| Эндпоинт       | Метод  | Описание                               |
|----------------|--------|----------------------------------------|
| /              | GET    | Информация о сервисе                   |
| /health        | GET    | Проверка работоспособности             |
| /docs          | GET    | Swagger документация                   |
| /metrics       | GET    | Метрики для Prometheus                 |
| /predict       | POST   | Предсказание для одного клиента        |
| /predict_batch | POST   | Массовое предсказание                  |


## Пример запроса к /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 2052.0
  }'
```

## Пример ответа
```json
{
  "prediction": 0,
  "probability": 0.3515,
  "threshold": 0.5,
  "class_label": "Не ушли",
  "timestamp": "2026-05-28T12:00:00.000000"
}
```

## Пример массового запроса (/predict_batch)
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type": application/json \
  -d '{
    "customers": [
      {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 2052.0
      },
      {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2160.0
      }
    ]
  }'
```

## Структура проекта

```
project/
├── README.md                 # Инструкция по запуску
├── report.md                 # Отчёт по проекту
├── self-checklist.md         # Чек-лист самопроверки
├── requirements.txt          # Зависимости Python
├── .env.example              # Шаблон переменных окружения
├── .gitignore                # Исключения для Git
├── .dockerignore             # Исключения для Docker
│
├── data/
│   ├── raw/                  # Исходные данные
│   └── processed/            # Обработанные данные
│
├── notebooks/                # Jupyter ноутбуки
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Advanced_Validation.ipynb
│
├── src/
│   └── service/
│       ├── app.py            # FastAPI приложение
│       └── model_loader.py   # Загрузчик модели
│
├── models/                   # Сохранённая модель и артефакты
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── columns_order.pkl
│   └── best_threshold.pkl
│
├── configs/
│   └── config.yaml           # Конфигурационный файл
│
├── tests/
│   ├── test_model.py         # Тест модели
│   └── test_api.py           # Тест API
│
└── docker/
    ├── Dockerfile            # Docker образ
    └── docker-compose.yml    # Docker Compose
```