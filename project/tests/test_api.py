import requests
import json

BASE_URL = "http://localhost:8000"

# Тестовые данные
test_data = {
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
}

print("=== ТЕСТ API ===")

# Тест health
print("\n1. Проверка /health...")
response = requests.get(f"{BASE_URL}/health")
print(f"   Статус: {response.status_code}")
print(f"   Ответ: {response.json()}")

# Тест predict
print("\n2. Проверка /predict...")
response = requests.post(f"{BASE_URL}/predict", json=test_data)
print(f"   Статус: {response.status_code}")
print(f"   Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

print("\n=== ТЕСТ ЗАВЕРШЁН ===")