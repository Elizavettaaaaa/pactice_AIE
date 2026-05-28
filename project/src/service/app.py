"""
FastAPI сервис для прогнозирования оттока клиентов
Customer Churn Prediction API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Импорт загрузчика модели
from .model_loader import model_loader

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus метрики
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP request latency', 
    ['method', 'endpoint']
)
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['prediction_class']
)

# Создание приложения
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API для прогнозирования оттока клиентов телекоммуникационной компании",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных для запросов и ответов


class CustomerFeatures(BaseModel):
    """
    Модель входных данных клиента
    """
    gender: str = Field(..., description="Пол (Male/Female)", example="Male")
    SeniorCitizen: int = Field(0, ge=0, le=1, description="Пенсионер (0/1)", example=0)
    Partner: str = Field(..., description="Наличие партнёра (Yes/No)", example="Yes")
    Dependents: str = Field(..., description="Наличие иждивенцев (Yes/No)", example="No")
    tenure: int = Field(..., ge=0, le=72, description="Длительность обслуживания (месяцы)", example=12)
    PhoneService: str = Field(..., description="Услуги телефонии (Yes/No)", example="Yes")
    MultipleLines: str = Field(..., description="Несколько линий (Yes/No/No phone service)", example="No")
    InternetService: str = Field(..., description="Интернет-услуги (DSL/Fiber optic/No)", example="DSL")
    OnlineSecurity: str = Field(..., description="Онлайн-безопасность (Yes/No/No internet service)", example="No")
    OnlineBackup: str = Field(..., description="Онлайн-резервное копирование (Yes/No/No internet service)", example="No")
    DeviceProtection: str = Field(..., description="Защита устройств (Yes/No/No internet service)", example="No")
    TechSupport: str = Field(..., description="Техподдержка (Yes/No/No internet service)", example="No")
    StreamingTV: str = Field(..., description="Стриминг ТВ (Yes/No/No internet service)", example="No")
    StreamingMovies: str = Field(..., description="Стриминг фильмов (Yes/No/No internet service)", example="No")
    Contract: str = Field(..., description="Тип контракта (Month-to-month/One year/Two year)", example="Month-to-month")
    PaperlessBilling: str = Field(..., description="Безбумажный счёт (Yes/No)", example="Yes")
    PaymentMethod: str = Field(..., description="Способ оплаты", example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, description="Ежемесячная плата", example=70.0)
    TotalCharges: float = Field(..., ge=0, description="Общая сумма платежей", example=840.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 840.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Модель ответа API
    """
    prediction: int = Field(..., description="Предсказанный класс (0 - не ушли, 1 - ушли)")
    probability: float = Field(..., description="Вероятность ухода (0-1)", ge=0, le=1)
    threshold: float = Field(..., description="Порог классификации")
    class_label: str = Field(..., description="Текстовое описание класса")
    timestamp: str = Field(..., description="Время предсказания")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "threshold": 0.5,
                "class_label": "Ушли",
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Модель массового запроса
    """
    customers: List[CustomerFeatures]


class HealthResponse(BaseModel):
    """
    Модель ответа health check
    """
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# Загрузка модели при старте
@app.on_event("startup")
async def startup_event():
    """
    Загрузка модели при запуске приложения
    """
    logger.info("Запуск API-сервиса прогнозирования оттока клиентов")
    try:
        model_loader.load()
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise


# Эндпоинты API


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Корневой эндпоинт
    """
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "predict": "POST /predict",
        "predict_batch": "POST /predict_batch"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка работоспособности сервиса
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/metrics")
async def get_metrics():
    """
    Метрики для Prometheus
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """
    Предсказание для одного клиента
    """
    start_time = time.time()
    
    try:
        # Преобразуем Pydantic модель в словарь
        data = features.model_dump()
        
        # Получаем предсказание
        result = model_loader.predict(data)
        
        # Обновляем метрики
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        PREDICTION_COUNT.labels(prediction_class=result['class']).inc()
        
        # Логирование
        logger.info(f"Prediction made: class={result['class']}, prob={result['probability']:.4f}")
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            threshold=result['threshold'],
            class_label=result['class'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании: {str(e)}"
        )
    
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict").observe(latency)


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Массовое предсказание для нескольких клиентов
    """
    start_time = time.time()
    
    try:
        results = []
        for customer in request.customers:
            data = customer.model_dump()
            result = model_loader.predict(data)
            results.append(PredictionResponse(
                prediction=result['prediction'],
                probability=result['probability'],
                threshold=result['threshold'],
                class_label=result['class'],
                timestamp=datetime.now().isoformat()
            ))
            PREDICTION_COUNT.labels(prediction_class=result['class']).inc()
        
        REQUEST_COUNT.labels(method="POST", endpoint="/predict_batch", status="200").inc()
        logger.info(f"Batch prediction made for {len(results)} customers")
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка при массовом предсказании: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict_batch", status="500").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании: {str(e)}"
        )
    
    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict_batch").observe(latency)