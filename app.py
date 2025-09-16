from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

#pre-trained model
model_file = 'decision_tree_model.pkl'
if not os.path.exists(model_file):
    print("Error: Model file not found. Please run train_model.py first.")
    exit()

with open(model_file, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sample_input = np.array([
        int(request.form['crop_id']),
        int(request.form['soil_type']),
        int(request.form['seedling_stage']),
        int(request.form['moisture']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['rainfall'])
    ]).reshape(1, -1)

    prediction = model.predict(sample_input)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    print("Flask app running on http://127.0.0.1:5000/")
    app.run(debug=True)
