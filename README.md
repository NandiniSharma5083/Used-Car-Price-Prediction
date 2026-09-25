# Old Car (Used Car) Price Prediction

A complete, runnable machine learning project that predicts the selling
price of old/used cars from features like age, mileage, brand, fuel type,
transmission, and ownership history.

## Project structure

```
car_price_prediction/
├── data/
│   ├── generate_data.py     # creates a synthetic 3,000-row used-car dataset
│   └── used_cars.csv        # generated dataset (output of the script above)
├── models/
│   ├── best_model.joblib        # trained pipeline (preprocessing + model)
│   ├── feature_columns.joblib   # feature column order, for correct inference
│   └── best_model_name.txt      # name of the winning model
├── outputs/                 # charts and metrics produced by the scripts
├── eda.py                   # exploratory data analysis + charts
├── train_model.py           # trains & compares 4 models, saves the best one
├── predict.py                # loads the saved model and predicts a price
├── app.py                    # Flask front-end around predict_price()
├── templates/
│   └── index.html            # prediction form UI
├── requirements.txt
└── README.md
```

## How to run

```bash
cd car_price_prediction
pip install -r requirements.txt

# 1. Generate the dataset (skip this if you plug in your own CSV instead)
python3 data/generate_data.py

# 2. Explore the data (saves 5 charts to outputs/)
python3 eda.py

# 3. Train & compare models, save the best one to models/
python3 train_model.py

# 4. Predict the price of a new car from the command line
python3 predict.py

# 5. Or launch the web frontend
python3 app.py
# then open http://127.0.0.1:5000
```

## Frontend

`app.py` is a small Flask app that wraps `predict_price()` from `predict.py`
in a form: pick brand, model, age, km driven, fuel type, transmission,
owner type, city, mileage, engine size, and seats, and it returns the
predicted selling price on the same page. It loads the trained pipeline
once at startup rather than per-request, and shows a friendly error
instead of crashing if a field is left invalid.

## Using a real dataset instead of synthetic data

Replace `data/used_cars.csv` with a real dataset (e.g. Kaggle's "Vehicle
Dataset from CarDekho" or "Used Car Price Prediction"), keeping these
column names — or edit the `features_num` / `features_cat` lists at the
top of `train_model.py` to match your columns:

| Column | Type | Description |
|---|---|---|
| `brand` | categorical | Manufacturer |
| `model` | categorical | Model name |
| `age` | numeric | Car age in years |
| `km_driven` | numeric | Total kilometers driven |
| `fuel_type` | categorical | Petrol / Diesel / CNG / Electric |
| `transmission` | categorical | Manual / Automatic |
| `owner_type` | categorical | First / Second / Third / Fourth & Above |
| `city` | categorical | Location of sale |
| `mileage_kmpl` | numeric | Fuel efficiency |
| `engine_cc` | numeric | Engine displacement |
| `seats` | numeric | Seating capacity |
| `selling_price` | numeric (target) | Price the car sold for |

## Approach

1. **Data generation** — a synthetic but realistic dataset is built with a
   ground-truth depreciation formula (≈12%/year decay, mileage penalty,
   ownership penalty, transmission/fuel premiums, plus market noise), so
   the whole pipeline can be demoed without external downloads.
2. **EDA** (`eda.py`) — distribution of prices, price vs. age, price vs.
   km driven, average price by brand, and a correlation heatmap.
3. **Preprocessing** — numeric features are standardized; categorical
   features are one-hot encoded, all inside an sklearn `ColumnTransformer`
   so the exact same transform is reapplied at prediction time.
4. **Modeling** — four regressors are trained and compared:
   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor

   The best model (by R² on a held-out 20% test set) is saved to
   `models/best_model.joblib`.
5. **Evaluation** — MAE, RMSE, and R² are reported for every model
   (`outputs/model_comparison.csv`), plus an actual-vs-predicted scatter
   plot and a feature-importance chart for the winning model.
6. **Prediction** (`predict.py`) — loads the saved pipeline and predicts
   the price for a new car description; `predict_price()` can be imported
   into a script, notebook, or a small Flask/Streamlit app.

## Results (synthetic dataset, this run)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | ~37,000 | ~62,000 | **0.989** |
| Random Forest | ~49,000 | ~83,000 | 0.980 |
| Ridge | ~209,000 | ~329,000 | 0.678 |
| Linear Regression | ~209,000 | ~329,000 | 0.678 |

Tree-based ensembles substantially outperform linear models here because
price depreciation is highly non-linear (exponential decay with
interaction effects between age, mileage, and ownership).

## Possible extensions

- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Log-transform the target (`selling_price`) to handle its right skew
- Add cross-validation instead of a single train/test split
- Try `XGBoost` / `LightGBM` for further accuracy gains
- Deploy `app.py` to Render/Railway for a live public demo
