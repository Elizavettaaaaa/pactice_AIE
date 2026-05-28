import joblib
import pandas as pd
import numpy as np

# Загружаем модель и артефакты
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
columns_order = joblib.load('models/columns_order.pkl')

# Тестовый клиент
test_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.5,
    'TotalCharges': 2052.0
}

# Преобразуем в DataFrame
df_input = pd.DataFrame([test_data])

# Категориальные колонки
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']

# One-Hot Encoding
df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

# Добавляем недостающие колонки
for col in columns_order:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Сортируем
df_encoded = df_encoded[columns_order]

# Масштабируем
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

# Предсказание
probability = model.predict_proba(df_encoded)[0, 1]
prediction = 1 if probability >= 0.5 else 0

print(f"Вероятность ухода: {probability:.4f}")
print(f"Предсказание: {'Ушли' if prediction == 1 else 'Не ушли'}")