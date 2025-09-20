import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def analyze_dataset():
    """Analyze the current dataset"""
    print("=== DATASET ANALYSIS ===")
    df = pd.read_csv("Irrigationdatset.csv")
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nTarget Distribution:")
    print(df['Result'].value_counts())
    
    print("\nCategorical Columns Analysis:")
    print("Crop types:", df['crop'].nunique(), "unique values")
    print("Crop samples:", df['crop'].value_counts().head())
    
    print("\nSoil types:", df['soil_type'].nunique(), "unique values")
    print("Soil samples:", df['soil_type'].value_counts())
    
    print("\nSeedling Stages:", df['Seedling Stage'].nunique(), "unique values")
    print("Stage samples:", df['Seedling Stage'].value_counts())
    
    print("\nNumerical Columns Statistics:")
    print(df[['Moisture', 'Temperature', 'Humidity', 'Rainfall']].describe())
    
    return df

def train_model_with_actual_data():
    """Train model with the actual dataset format"""
    print("\n=== TRAINING MODEL WITH ACTUAL DATA ===")
    
    # Load dataset
    df = pd.read_csv("Irrigationdatset.csv")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Handle missing values if any
    if df.isnull().sum().any():
        print("Handling missing values...")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_stage = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df['crop'])
    df['soil_type_encoded'] = le_soil.fit_transform(df['soil_type'])
    df['seedling_stage_encoded'] = le_stage.fit_transform(df['Seedling Stage'])
    
    # Prepare features and target
    feature_columns = ['crop_encoded', 'soil_type_encoded', 'seedling_stage_encoded', 
                      'Moisture', 'Temperature', 'Humidity', 'Rainfall']
    
    X = df[feature_columns]
    y = df['Result']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Decision Tree
    print("\nTraining Decision Tree Classifier...")
    model = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=15)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and encoders
    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    encoders = {
        'crop_encoder': le_crop,
        'soil_encoder': le_soil,
        'stage_encoder': le_stage
    }
    
    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    
    print(f"\nModel saved as 'decision_tree_model.pkl'")
    print(f"Encoders saved as 'label_encoders.pkl'")
    
    return model, accuracy, encoders

def create_sample_dataset_with_desired_columns():
    """Create a sample dataset with the column names you want"""
    print("\n=== CREATING SAMPLE DATASET WITH DESIRED COLUMNS ===")
    
    # Create sample data with your desired column names
    sample_data = {
        'crop': ['Rice', 'Wheat', 'Corn', 'Tomato', 'Potato', 'Rice', 'Wheat', 'Corn', 'Tomato', 'Potato'],
        'soil_type': ['Clay', 'Sandy', 'Loamy', 'Clay', 'Sandy', 'Loamy', 'Clay', 'Sandy', 'Loamy', 'Clay'],
        'Seedling Stage': ['Early', 'Mid', 'Late', 'Early', 'Mid', 'Late', 'Early', 'Mid', 'Late', 'Early'],
        'Moisture (min. % of moisture in soil for healthy growth of crops': [45, 60, 30, 55, 40, 65, 50, 35, 45, 60],
        'Temperature': [25.5, 28.0, 22.0, 30.0, 26.5, 27.5, 24.0, 29.0, 25.0, 28.5],
        'Humidity': [70, 65, 80, 60, 75, 68, 72, 63, 70, 65],
        'Rainfall in mm (based on humidity and temperature': [5.2, 0.0, 12.5, 2.1, 8.0, 0.5, 15.0, 1.2, 3.5, 0.8],
        'Result (will rain or not within 2 hrs)': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_dataset_with_desired_columns.csv', index=False)
    
    print("Sample dataset created with your desired column names:")
    print("File: 'sample_dataset_with_desired_columns.csv'")
    print("\nSample data:")
    print(df_sample)
    
    return df_sample

if __name__ == "__main__":
    print("=== AGRICULTURAL IRRIGATION MODEL TRAINING ===")
    
    # Analyze current dataset
    df_analysis = analyze_dataset()
    
    # Train with actual data
    model, accuracy, encoders = train_model_with_actual_data()
    
    # Create sample dataset with desired column names
    sample_df = create_sample_dataset_with_desired_columns()
    
    print(f"\n✅ Training completed successfully!")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Model is ready for use in the Flask application!")
