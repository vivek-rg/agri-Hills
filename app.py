from flask import Flask, request, render_template
import numpy as np
import pickle
import os
import requests  # <-- for API call

app = Flask(__name__)

# Pre-trained model
model_file = 'decision_tree_model.pkl'
if not os.path.exists(model_file):
    print("Error: Model file not found. Please run train_model.py first.")
    exit()

with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Your OpenWeather API Key
API_KEY = "f4ccd28ceb577a556911dfd947e0ab84"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    crop_id = int(request.form['crop_id'])
    soil_type = int(request.form['soil_type'])
    seedling_stage = int(request.form['seedling_stage'])
    moisture = int(request.form['moisture'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    rainfall = float(request.form['rainfall'])
    location = request.form.get('location', 'Bangalore')  # default city

    # Prepare data for model
    sample_input = np.array([
        crop_id, soil_type, seedling_stage,
        moisture, temperature, humidity, rainfall
    ]).reshape(1, -1)

    prediction = model.predict(sample_input)[0]

    # 🌦️ Check forecast for next 3 hours using OpenWeather
    rain_forecast = False
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if "list" in data:
            next_3h = data["list"][0]  # first forecast = next 3 hours
            if "rain" in next_3h and next_3h["rain"].get("3h", 0) > 0:
                rain_forecast = True
    except Exception as e:
        print("Weather API error:", e)

    return render_template(
        'result.html',
        prediction=prediction,
        rainfall=rainfall,
        rain_forecast=rain_forecast
    )

if __name__ == '__main__':
    print("Flask app running on http://127.0.0.1:5000/")
    app.run(debug=True)
