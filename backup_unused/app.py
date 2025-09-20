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

    # Calculate threshold moisture
    threshold_moisture, moisture_advice = calculate_threshold_moisture(
        crop_id, soil_type, seedling_stage, temperature, humidity
    )

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
        rain_forecast=rain_forecast,
        threshold_moisture=threshold_moisture,
        current_moisture=moisture,
        moisture_advice=moisture_advice
    )

def calculate_threshold_moisture(crop_id, soil_type, seedling_stage, temperature, humidity):
    """Calculate optimal moisture threshold based on environmental factors"""
    
    # Crop-specific moisture requirements (base thresholds)
    crop_requirements = {
        0: 40,  # CARROT - Root crop
        1: 35,  # CHILLI - Vegetable
        2: 45,  # POTATO - Root crop
        3: 30,  # TOMATO - Fruit crop
        4: 25   # WHEAT - Grain crop
    }
    
    # Soil type moisture retention factors
    soil_factors = {
        0: 1.2,  # Alluvial - good retention
        1: 1.3,  # Black - excellent retention
        2: 0.8,  # Chalky - poor retention
        3: 1.4,  # Clay - excellent retention
        4: 1.1,  # Loam - good retention
        5: 0.9,  # Red - moderate retention
        6: 0.7   # Sandy - poor retention
    }
    
    # Growth stage moisture multipliers
    stage_multipliers = {
        0: 1.2,  # Flowering - needs more water
        1: 1.3,  # Fruit/Grain/Bulb Formation - critical stage
        2: 1.0,  # Germination - moderate needs
        3: 0.8,  # Harvest - less water needed
        4: 1.1   # Maturation - moderate needs
    }
    
    # Get base moisture requirement
    base_moisture = crop_requirements.get(crop_id, 30)
    
    # Temperature adjustment (higher temp = more moisture needed)
    temp_factor = 1 + (temperature - 25) * 0.02  # 2% increase per degree above 25°C
    
    # Humidity adjustment (lower humidity = more moisture needed)
    humidity_factor = 1 + (50 - humidity) * 0.01  # 1% increase per % below 50% humidity
    
    # Soil type factor
    soil_factor = soil_factors.get(soil_type, 1.0)
    
    # Growth stage factor
    stage_factor = stage_multipliers.get(seedling_stage, 1.0)
    
    # Calculate final threshold
    threshold = base_moisture * temp_factor * humidity_factor * soil_factor * stage_factor
    
    # Ensure threshold is within reasonable bounds (15-70%)
    threshold = max(15, min(70, threshold))
    
    # Round to 1 decimal place
    threshold = round(threshold, 1)
    
    # Generate advice based on threshold
    if threshold > 50:
        advice = "High moisture requirement due to hot, dry conditions and crop needs."
    elif threshold > 35:
        advice = "Moderate moisture requirement for optimal crop growth."
    else:
        advice = "Lower moisture requirement due to cool, humid conditions."
    
    return threshold, advice

if __name__ == '__main__':
    print("Flask app running on http://127.0.0.1:5000/")
    app.run(debug=True)
