"""
app.py
------
Flask front-end around predict.py's predict_price(). Serves a form
where the user enters a car's details and gets back the predicted
selling price.

Run:
  python3 app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request

from predict import load_model, predict_price

app = Flask(__name__)

# Load once at startup instead of on every request
_, FEATURE_COLUMNS = load_model()

BRANDS_MODELS = {
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Alto", "Dzire"],
    "Hyundai": ["i20", "Creta", "Venue", "i10", "Verna"],
    "Tata": ["Nexon", "Tiago", "Altroz", "Harrier"],
    "Honda": ["City", "Amaze", "Jazz"],
    "Toyota": ["Innova", "Fortuner", "Glanza"],
    "Mahindra": ["XUV500", "Scorpio", "Bolero"],
    "Ford": ["EcoSport", "Figo"],
}
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "Electric"]
TRANSMISSIONS = ["Manual", "Automatic"]
OWNER_TYPES = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Kolkata", "Chandigarh"]

DEFAULTS = {
    "age": 5, "km_driven": 45000, "mileage_kmpl": 18.5, "engine_cc": 1200,
    "seats": 5, "brand": "Hyundai", "model": "i20", "fuel_type": "Petrol",
    "transmission": "Manual", "owner_type": "First Owner", "city": "Pune",
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_values = DEFAULTS.copy()

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            car = {
                "age": int(request.form["age"]),
                "km_driven": int(request.form["km_driven"]),
                "mileage_kmpl": float(request.form["mileage_kmpl"]),
                "engine_cc": int(request.form["engine_cc"]),
                "seats": int(request.form["seats"]),
                "brand": request.form["brand"],
                "model": request.form["model"],
                "fuel_type": request.form["fuel_type"],
                "transmission": request.form["transmission"],
                "owner_type": request.form["owner_type"],
                "city": request.form["city"],
            }
            prediction = predict_price(car)
        except Exception as exc:  # noqa: BLE001 - surface any bad input to the form
            error = str(exc)

    return render_template(
        "index.html",
        brands_models=BRANDS_MODELS,
        fuel_types=FUEL_TYPES,
        transmissions=TRANSMISSIONS,
        owner_types=OWNER_TYPES,
        cities=CITIES,
        values=form_values,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
