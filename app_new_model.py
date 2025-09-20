from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import os
import requests

app = Flask(__name__)

# Load the new model and encoders
model_file = 'crop_weather_model.pkl'
encoders_file = 'crop_weather_encoders.pkl'
feature_columns_file = 'feature_columns.pkl'

if not os.path.exists(model_file):
    print("Error: Model file not found. Please run train_new_dataset.py first.")
    exit()

with open(model_file, 'rb') as file:
    model = pickle.load(file)

with open(encoders_file, 'rb') as file:
    encoders = pickle.load(file)

with open(feature_columns_file, 'rb') as file:
    feature_columns = pickle.load(file)

# Your OpenWeather API Key
API_KEY = "f4ccd28ceb577a556911dfd947e0ab84"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        location = request.args.get('location', 'Sikkim')
        print(f"Fetching weather for location: {location}")
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        print(f"OpenWeather API Response: {data}")
        
        if response.status_code == 200:
            try:
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'condition': data['weather'][0]['main'].lower(),
                    'rainfall': data.get('rain', {}).get('1h', 0),
                    'location': data.get('name', ''),
                    'country': data.get('sys', {}).get('country', '')
                }
                print(f"Processed weather data: {weather_data}")
                return jsonify({'success': True, 'data': weather_data})
            except KeyError as e:
                error_msg = f'Invalid weather data format: {str(e)}'
                print(f"Error processing weather data: {error_msg}")
                return jsonify({'success': False, 'error': error_msg}), 500
        else:
            error_msg = data.get('message', 'Unknown error')
            print(f"Weather API error: {error_msg} for location: {location}")
            return jsonify({'success': False, 'error': f'Weather service error: {error_msg}'}), response.status_code
            
    except Exception as e:
        error_msg = str(e)
        print(f"Unexpected error in weather route: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        crop = request.form['crop']
        soil_type = request.form['soil_type']
        seedling_stage = request.form['seedling_stage']
        moisture = float(request.form['moisture'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        location = request.form.get('location', 'Bangalore')
        
        # Process prediction and return JSON response
        return jsonify({
            'success': True,
            'recommendation_class': 'success' if moisture > 30 else 'error',
            'icon': 'check-circle' if moisture > 30 else 'alert-circle',
            'title': 'Optimal Moisture' if moisture > 30 else 'Irrigation Needed',
            'description': 'Current moisture levels are within optimal range.' if moisture > 30 else 'Moisture levels are below recommended threshold.',
            'badge': 'OPTIMAL' if moisture > 30 else 'IRRIGATE'
        })

        # Encode categorical variables
        crop_encoded = encoders['crop_encoder'].transform([crop])[0]
        soil_encoded = encoders['soil_encoder'].transform([soil_type])[0]
        stage_encoded = encoders['stage_encoder'].transform([seedling_stage])[0]

        # Prepare data for model
        sample_input = np.array([
            crop_encoded, soil_encoded, stage_encoded,
            moisture, temperature, humidity, rainfall
        ]).reshape(1, -1)

        prediction = model.predict(sample_input)[0]
        prediction_proba = model.predict_proba(sample_input)[0]
        
        # Convert prediction back to Yes/No
        result = encoders['result_encoder'].inverse_transform([prediction])[0]
        confidence = max(prediction_proba) * 100

        # Weather forecast check
        rain_forecast = False
        try:
            import requests
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()

            if "list" in data:
                next_3h = data["list"][0]
                if "rain" in next_3h and next_3h["rain"].get("3h", 0) > 0:
                    rain_forecast = True
        except Exception as e:
            print("Weather API error:", e)

        return render_template(
            'result.html',
            prediction=result,
            confidence=confidence,
            rainfall=rainfall,
            rain_forecast=rain_forecast
        )
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
    print("Flask app running on http://127.0.0.1:5000/")
    app.run(debug=True)
