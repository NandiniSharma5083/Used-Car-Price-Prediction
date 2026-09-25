"""
predict.py
----------
Loads the trained model and predicts the selling price for a new
(hypothetical) used car. Edit the `sample_car` dict below, or import
`predict_price()` into your own code / a small web app.

Run:
  python3 predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"
FEATURES_PATH = "models/feature_columns.joblib"


def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def predict_price(car: dict) -> float:
    """
    car: dict with keys:
      age (int), km_driven (int), mileage_kmpl (float), engine_cc (int),
      seats (int), brand (str), model (str), fuel_type (str),
      transmission (str), owner_type (str), city (str)
    """
    model, feature_columns = load_model()
    row = pd.DataFrame([car])[feature_columns]
    price = model.predict(row)[0]
    return round(float(price), -2)


if __name__ == "__main__":
    sample_car = {
        "age": 5,
        "km_driven": 45000,
        "mileage_kmpl": 18.5,
        "engine_cc": 1200,
        "seats": 5,
        "brand": "Hyundai",
        "model": "i20",
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "owner_type": "First Owner",
        "city": "Pune",
    }

    predicted = predict_price(sample_car)
    print("Sample car:")
    for k, v in sample_car.items():
        print(f"  {k}: {v}")
    print(f"\nPredicted selling price: {predicted:,.0f}")
