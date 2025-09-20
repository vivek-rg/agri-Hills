import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def analyze_new_dataset():
    """Analyze the new crop weather dataset"""
    print("=== NEW DATASET ANALYSIS ===")
    df = pd.read_csv("crop_weather_dataset.csv")
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nTarget Distribution:")
    target_counts = df['Result (will rain or not within 2 hrs)'].value_counts()
    print(target_counts)
    print(f"Class balance: {target_counts['Yes']/len(df)*100:.1f}% Yes, {target_counts['No']/len(df)*100:.1f}% No")
    
    print("\nCategorical Variables Analysis:")
    print(f"Crop types: {df['crop'].nunique()} unique values")
    print("Top 10 crops:")
    print(df['crop'].value_counts().head(10))
    
    print(f"\nSoil types: {df['soil_type'].nunique()} unique values")
    print(df['soil_type'].value_counts())
    
    print(f"\nSeedling Stages: {df['Seedling Stage'].nunique()} unique values")
    print(df['Seedling Stage'].value_counts())
    
    print("\nNumerical Variables Statistics:")
    numerical_cols = ['Moisture (min. % of moisture in soil for healthy growth of crops', 
                     'Temperature', 'Humidity', 'Rainfall in mm (based on humidity and temperature']
    print(df[numerical_cols].describe())
    
    return df

def train_model_with_new_dataset():
    """Train model with the new crop weather dataset"""
    print("\n=== TRAINING MODEL WITH NEW DATASET ===")
    
    # Load dataset
    df = pd.read_csv("crop_weather_dataset.csv")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Check for missing values
    if df.isnull().sum().any():
        print("Handling missing values...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    else:
        print("No missing values found.")
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_stage = LabelEncoder()
    le_result = LabelEncoder()
    
    # Encode categorical features
    df['crop_encoded'] = le_crop.fit_transform(df['crop'])
    df['soil_type_encoded'] = le_soil.fit_transform(df['soil_type'])
    df['seedling_stage_encoded'] = le_stage.fit_transform(df['Seedling Stage'])
    
    # Encode target variable (Yes/No to 1/0)
    df['result_encoded'] = le_result.fit_transform(df['Result (will rain or not within 2 hrs)'])
    
    print(f"Crop encoding: {dict(zip(le_crop.classes_, le_crop.transform(le_crop.classes_)))}")
    print(f"Soil encoding: {dict(zip(le_soil.classes_, le_soil.transform(le_soil.classes_)))}")
    print(f"Stage encoding: {dict(zip(le_stage.classes_, le_stage.transform(le_stage.classes_)))}")
    print(f"Result encoding: {dict(zip(le_result.classes_, le_result.transform(le_result.classes_)))}")
    
    # Prepare features and target
    feature_columns = ['crop_encoded', 'soil_type_encoded', 'seedling_stage_encoded', 
                      'Moisture (min. % of moisture in soil for healthy growth of crops', 
                      'Temperature', 'Humidity', 'Rainfall in mm (based on humidity and temperature']
    
    X = df[feature_columns]
    y = df['result_encoded']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Decision Tree
    print("\nTraining Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=15)
    dt_model.fit(X_train, y_train)
    
    # Train Random Forest for comparison
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=15)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate metrics for both models
    dt_accuracy = accuracy_score(y_test, dt_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Choose the best model
    if dt_accuracy >= rf_accuracy:
        best_model = dt_model
        best_name = "Decision Tree"
        best_accuracy = dt_accuracy
    else:
        best_model = rf_model
        best_name = "Random Forest"
        best_accuracy = rf_accuracy
    
    print(f"\nBest model: {best_name} with accuracy {best_accuracy:.4f}")
    
    # Detailed performance metrics
    print(f"\n{best_name} Classification Report:")
    if best_name == "Decision Tree":
        print(classification_report(y_test, dt_pred, target_names=['No', 'Yes']))
        print(f"\n{best_name} Confusion Matrix:")
        print(confusion_matrix(y_test, dt_pred))
    else:
        print(classification_report(y_test, rf_pred, target_names=['No', 'Yes']))
        print(f"\n{best_name} Confusion Matrix:")
        print(confusion_matrix(y_test, rf_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{best_name} Feature Importance:")
    print(feature_importance)
    
    # Save model and encoders
    with open('crop_weather_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    
    encoders = {
        'crop_encoder': le_crop,
        'soil_encoder': le_soil,
        'stage_encoder': le_stage,
        'result_encoder': le_result
    }
    
    with open('crop_weather_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    
    # Save feature column names
    with open('feature_columns.pkl', 'wb') as file:
        pickle.dump(feature_columns, file)
    
    print(f"\nModel saved as 'crop_weather_model.pkl'")
    print(f"Encoders saved as 'crop_weather_encoders.pkl'")
    print(f"Feature columns saved as 'feature_columns.pkl'")
    
    return best_model, best_accuracy, encoders, feature_columns

def create_updated_app():
    """Create an updated Flask app that works with the new model"""
    print("\n=== CREATING UPDATED FLASK APP ===")
    
    app_code = '''from flask import Flask, request, render_template
import numpy as np
import pickle
import os

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
    print("Flask app running on http://127.0.0.1:5000/")
    app.run(debug=True)
'''
    
    with open('app_new_model.py', 'w') as file:
        file.write(app_code)
    
    print("Updated Flask app created as 'app_new_model.py'")

if __name__ == "__main__":
    print("=== AGRICULTURAL CROP WEATHER MODEL TRAINING ===")
    
    # Analyze new dataset
    df_analysis = analyze_new_dataset()
    
    # Train model with new dataset
    model, accuracy, encoders, feature_columns = train_model_with_new_dataset()
    
    # Create updated Flask app
    create_updated_app()
    
    print(f"\n✅ Training completed successfully!")
    print(f"Best model accuracy: {accuracy:.4f}")
    print(f"Model is ready for use!")
    print(f"\nTo run the updated app: python app_new_model.py")
