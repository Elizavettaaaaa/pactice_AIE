"""
Модуль загрузки модели для API-сервиса
"""

import joblib
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelLoader:
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            current_dir = Path(__file__).parent
            self.model_dir = current_dir.parent.parent / 'models'
        else:
            self.model_dir = Path(model_dir)
        
        self.model = None
        self.scaler = None
        self.columns_order = None
        self.threshold = 0.5
        self.categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        
    def load(self) -> None:
        model_path = self.model_dir / 'churn_model.pkl'
        scaler_path = self.model_dir / 'scaler.pkl'
        columns_path = self.model_dir / 'columns_order.pkl'
        threshold_path = self.model_dir / 'best_threshold.pkl'
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.columns_order = joblib.load(columns_path)
        
        if threshold_path.exists():
            self.threshold = joblib.load(threshold_path)
        
        print(f"Модель загружена. Признаков: {len(self.columns_order)}")
    
    def predict(self, data: dict) -> dict:
        if self.model is None:
            self.load()
        
        # Создаём DataFrame
        df_input = pd.DataFrame([data])
        
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df_input, columns=self.categorical_cols, drop_first=True)
        
        # Добавляем недостающие колонки
        for col in self.columns_order:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Сортируем колонки
        df_encoded = df_encoded[self.columns_order]
        
        # Масштабируем числовые признаки
        df_encoded[self.numeric_cols] = self.scaler.transform(df_encoded[self.numeric_cols])
        
        # Предсказание
        probability = float(self.model.predict_proba(df_encoded)[0, 1])
        prediction = int(probability >= self.threshold)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'threshold': self.threshold,
            'class': 'Ушли' if prediction == 1 else 'Не ушли'
        }


model_loader = ModelLoader()