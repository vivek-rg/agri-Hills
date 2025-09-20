import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("=== Setting up the Agricultural Model ===")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"c:\Users\vivek\OneDrive\Desktop\sih\agri-Hills\crop_weather_dataset.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}")

# Encode categorical variables
print("\nEncoding categorical variables...")
encoders = {}
encoders['crop_encoder'] = LabelEncoder()
encoders['soil_encoder'] = LabelEncoder()
encoders['stage_encoder'] = LabelEncoder()
encoders['result_encoder'] = LabelEncoder()

# Encode features
df['crop_encoded'] = encoders['crop_encoder'].fit_transform(df['crop'])
df['soil_type_encoded'] = encoders['soil_encoder'].fit_transform(df['soil_type'])
df['seedling_stage_encoded'] = encoders['stage_encoder'].fit_transform(df['Seedling Stage'])
df['result_encoded'] = encoders['result_encoder'].fit_transform(df['Result (will rain or not within 2 hrs)'])

# Prepare features
feature_columns = [
    'crop_encoded', 'soil_type_encoded', 'seedling_stage_encoded',
    'Moisture (min. % of moisture in soil for healthy growth of crops',
    'Temperature', 'Humidity', 'Rainfall in mm (based on humidity and temperature'
]

X = df[feature_columns]
y = df['result_encoded']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train model
print("\nTraining Decision Tree Classifier...")
model = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=10)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and encoders
print("\nSaving model and encoders...")
with open('crop_weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as crop_weather_model.pkl")

with open('crop_weather_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("Encoders saved as crop_weather_encoders.pkl")

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("Feature columns saved as feature_columns.pkl")

print("\nSetup complete! You can now run the application.")