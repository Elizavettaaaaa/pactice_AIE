# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

## 1. Паспорт проекта

- **Название проекта:** Прогнозирование оттока клиентов (Customer Churn Prediction)
- **Автор:** Горохова Елизавета Сергеевна
- **Группа:** УИБО-01-22
- **Контакт:** @ellizaveeetttaaa

- **Краткое описание (2-4 предложения):**  
  Проект посвящён построению сервиса прогнозирования оттока клиентов телекоммуникационной компании. Используется открытый датасет Telco Customer Churn, модели машинного обучения (Logistic Regression, Random Forest, XGBoost, CatBoost). Результат – REST API на FastAPI, который по признакам клиента возвращает вероятность ухода и предсказанный класс.

---

## 2. Структура проекта

Проект организован в следующей структуре:

- `requirements.txt` – зависимости проекта (библиотеки Python, необходимые для запуска).
- `report.md` – отчёт по проекту (постановка задачи, данные, эксперименты, результаты).
- `self-checklist.md` – чеклист самопроверки проекта перед сдачей.
- `notebooks/` – экспериментальные ноутбуки:
  - `01_Telco_EDA_and_Preprocessing.ipynb` – разведочный анализ данных и предобработка.
  - `02_Telco_Model_Training.ipynb` – обучение и сравнение моделей.
  - `03_Advanced_Validation_and_Balancing.ipynb` – временная валидация и SMOTE.
- `src/` – основной код проекта:
  - `src/service/app.py` – FastAPI сервис.
  - `src/service/model_loader.py` – загрузчик модели и артефактов.
- `data/` – демонстрационные/учебные данные:
  - `data/raw/telco_churn.csv` – исходный датасет.
  - `data/processed/telco_cleaned.csv` – обработанные данные.
- `configs/` – конфигурационные файлы:
  - `config.yaml` – основной конфигурационный файл.
  - `README.md` – описание конфигов.
- `tests/` – тесты:
  - `test_model.py` – тест загрузки модели.
  - `test_api.py` – тест API (требует запущенного сервиса).
- `docker/` – контейнеризация:
  - `Dockerfile` – Docker образ.
  - `docker-compose.yml` – оркестрация контейнеров.
- `models/` – сохранённые модели и артефакты:
  - `churn_model.pkl` – финальная модель XGBoost.
  - `scaler.pkl` – StandardScaler.
  - `columns_order.pkl` – порядок колонок.
  - `best_threshold.pkl` – оптимальный порог классификации.

---

## 3. Требования и установка

### 3.1. Требования

- Python >= 3.12
- Docker (опционально, для контейнеризации)

### 3.2. Установка окружения

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Запуск обучения модели (эксперименты)

Эксперименты выполняются в Jupyter ноутбуках:

```bash
cd project/notebooks
jupyter notebook
```

Откройте последовательно:
1. `01_Telco_EDA_and_Preprocessing.ipynb`
2. `02_Telco_Model_Training.ipynb`
3. `03_Advanced_Validation_and_Balancing.ipynb`

### 4.2. Запуск сервиса (API)

```bash
cd project
source .venv/bin/activate
uvicorn src.service.app:app --reload --host 0.0.0.0 --port 8000
```

**Доступные эндпоинты:**
- `GET /` – информация о сервисе
- `GET /health` – проверка работоспособности
- `GET /docs` – Swagger документация
- `GET /metrics` – метрики Prometheus
- `POST /predict` – предсказание для одного клиента
- `POST /predict_batch` – массовое предсказание

**Проверка работоспособности:**

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:
```json
{"status":"healthy","model_loaded":true,"timestamp":"...","version":"1.0.0"}
```

### 4.3. Запуск через Docker

```bash
cd project
docker build -f docker/Dockerfile -t churn-prediction-api .
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
```

Или через docker-compose:

```bash
cd project/docker
docker-compose up -d
```

---

## 5. Данные

- **Источник:** открытый датасет Telco Customer Churn (Kaggle)
- **Ссылка:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Файлы в репозитории:**
- `data/raw/telco_churn.csv` – исходный датасет
- `data/processed/telco_cleaned.csv` – обработанные данные после EDA

Полную версию датасета можно скачать с Kaggle по ссылке выше.

---

## 6. Тесты

```bash
cd project
source .venv/bin/activate

python tests/test_model.py
python tests/test_api.py
```

---

## 7. Демонстрация на защите

На защите я:

1. Покажу структуру проекта (README.md, notebooks/, src/service/).
2. Запущу сервис через uvicorn и продемонстрирую:
   - Swagger UI (/docs) с описанием всех эндпоинтов
   - Запрос к /health
   - Запрос к /predict с примером клиента, получение JSON ответа
3. Покажу ноутбук 02 с результатами обучения моделей и таблицей сравнения метрик (F1-score = 0.635, Recall = 80.5%).
4. Покажу Docker – соберу образ и запущу контейнер с сервисом.

---

## 8. Ограничения и дальнейшая работа

**Текущие ограничения:**
- Модель обучена только на статических исторических данных
- Precision (52.4%) относительно низкий из-за приоритета Recall

**Направления дальнейшего развития:**
- Сбор дополнительных поведенческих признаков
- A/B-тестирование перед внедрением
- Мониторинг дрейфа модели (PSI)
- Эксперименты с методами балансировки SMOTE, ADASYN

---

## 9. Оценка проекта

По чеклисту self-checklist.md выполнено 10 из 10 пунктов.

| № | Критерий | Статус |
|---|----------|--------|
| 1 | Запуск сервиса | ✅ |
| 2 | Реальная модель в /predict | ✅ |
| 3 | EDA и эксперимент | ✅ |
| 4 | Сравнение моделей | ✅ |
| 5 | Структура кода | ✅ |
| 6 | Развёртывание (Docker) | ✅ |
| 7 | Конфиги и секреты | ✅ |
| 8 | Наблюдаемость | ✅ |
| 9 | Обоснование модели | ✅ |
| 10 | Демо-сценарий | ✅ |

**Предварительная оценка: 5 (отлично)**