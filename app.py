from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ---------------------------
# Load Model & Scaler
# ---------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("minmaxscaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Crop Label Mapping (22 crops)
# ---------------------------
crop_mapping = {
    0: "rice",
    1: "maize",
    2: "chickpea",
    3: "kidneybeans",
    4: "pigeonpeas",
    5: "mothbeans",
    6: "mungbean",
    7: "blackgram",
    8: "lentil",
    9: "pomegranate",
    10: "banana",
    11: "mango",
    12: "grapes",
    13: "watermelon",
    14: "muskmelon",
    15: "apple",
    16: "orange",
    17: "papaya",
    18: "coconut",
    19: "cotton",
    20: "jute",
    21: "coffee"
}


# ---------------------------
# Home Route
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# Predict Route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read Form Values
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        temp_unit = request.form["temp_unit"]  # "C" or "F"
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Convert Fahrenheit → Celsius
        if temp_unit == "F":
            temperature = (temperature - 32) * 5/9

        # Prepare input array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Predict encoded label
        prediction_encoded = model.predict(input_scaled)[0]

        # Convert encoded number → crop name
        prediction = crop_mapping[int(prediction_encoded)]

        return render_template("index.html", prediction_text=f"Recommended Crop: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)

