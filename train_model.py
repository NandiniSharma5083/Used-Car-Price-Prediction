"""
train_model.py
----------------
Trains and compares 4 regression models to predict used-car selling
price, evaluates each on a held-out test set, and saves the best
pipeline (preprocessing + model) to models/best_model.joblib.

Run:
  python3 train_model.py
"""

import joblib
import pandas as pd
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/used_cars.csv"
OUTPUT_DIR = Path("models/outputs")
TARGET = "selling_price"

features_num = ["age", "km_driven", "mileage_kmpl", "engine_cc", "seats"]
features_cat = ["brand", "model", "fuel_type", "transmission", "owner_type", "city"]
feature_columns = features_num + features_cat


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df[feature_columns]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted[name] = pipe
        print(f"{name:>18s}  MAE={mae:>10,.0f}  RMSE={rmse:>10,.0f}  R2={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name} (R2={results_df.iloc[0]['R2']:.4f})")

    joblib.dump(best_pipe, "models/best_model.joblib")
    joblib.dump(feature_columns, "models/feature_columns.joblib")
    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    # Actual vs predicted scatter for the winning model
    preds_best = best_pipe.predict(X_test)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, preds_best, alpha=0.4, color="steelblue")
    lims = [min(y_test.min(), preds_best.min()), max(y_test.max(), preds_best.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=120)
    plt.close()

    # Feature importance (only for tree-based models)
    model_obj = best_pipe.named_steps["model"]
    if hasattr(model_obj, "feature_importances_"):
        ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(features_cat))
        all_names = features_num + cat_names
        importances = pd.Series(model_obj.feature_importances_, index=all_names)
        importances = importances.sort_values(ascending=False).head(15)

        plt.figure(figsize=(8, 6))
        importances[::-1].plot(kind="barh", color="darkorange")
        plt.title(f"Top 15 Feature Importances ({best_name})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=120)
        plt.close()

    print("\nSaved: models/best_model.joblib, models/feature_columns.joblib,")
    print("       models/outputs/model_comparison.csv, models/outputs/actual_vs_predicted.png"
          + (", models/outputs/feature_importance.png" if hasattr(model_obj, "feature_importances_") else ""))


if __name__ == "__main__":
    main()
