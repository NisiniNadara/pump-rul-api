import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# load AI model
model = joblib.load("rul_model_final.pkl")
scaler = joblib.load("rul_scaler_final.pkl")

selected_sensors = [
'sensor_02',
'sensor_03',
'sensor_07',
'sensor_22',
'sensor_23',
'sensor_25',
'sensor_26',
'sensor_29',
'sensor_34'
]

@app.route("/")
def home():
    return "Pump RUL Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_df = pd.DataFrame([data], columns=selected_sensors)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return jsonify({"predicted_rul": float(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)