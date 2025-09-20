"""
Data Preparation Guide for Agricultural Irrigation Dataset

This script helps you prepare new data in the correct format for training.
"""

import pandas as pd
import numpy as np

def create_sample_new_data():
    """
    Create a sample new dataset with the correct format
    """
    
    # Sample data structure
    sample_data = {
        'crop': ['Rice', 'Wheat', 'Corn', 'Tomato', 'Potato', 'Rice', 'Wheat', 'Corn'],
        'soil_type': ['Clay', 'Sandy', 'Loamy', 'Clay', 'Sandy', 'Loamy', 'Clay', 'Sandy'],
        'Seedling Stage': ['Early', 'Mid', 'Late', 'Early', 'Mid', 'Late', 'Early', 'Mid'],
        'moisture': [45, 60, 30, 55, 40, 65, 50, 35],
        'temperature': [25.5, 28.0, 22.0, 30.0, 26.5, 27.5, 24.0, 29.0],
        'humidity': [70, 65, 80, 60, 75, 68, 72, 63],
        'rainfall': [5.2, 0.0, 12.5, 2.1, 8.0, 0.5, 15.0, 1.2],
        'Result': ['Irrigate', 'No Irrigation', 'Irrigate', 'No Irrigation', 'Irrigate', 'No Irrigation', 'Irrigate', 'No Irrigation']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_new_data.csv', index=False)
    print("Sample new data created as 'sample_new_data.csv'")
    print("\nSample data:")
    print(df)
    
    return df

def validate_data_format(file_path):
    """
    Validate if the data is in the correct format
    """
    try:
        df = pd.read_csv(file_path)
        
        required_columns = ['crop', 'soil_type', 'Seedling Stage', 'moisture', 'temperature', 'humidity', 'rainfall', 'Result']
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns found: {list(df.columns)}")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Check data types
        print("\nData types:")
        print(df.dtypes)
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Check target variable values
        if 'Result' in df.columns:
            print(f"\nTarget variable 'Result' unique values:")
            print(df['Result'].value_counts())
        
        return True
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def prepare_your_data():
    """
    Instructions for preparing your own data
    """
    print("""
    📋 DATA PREPARATION INSTRUCTIONS
    ================================
    
    Your dataset should be a CSV file with the following columns:
    
    1. crop (string): Type of crop (e.g., 'Rice', 'Wheat', 'Corn', 'Tomato', 'Potato')
    2. soil_type (string): Type of soil (e.g., 'Clay', 'Sandy', 'Loamy')
    3. Seedling Stage (string): Growth stage (e.g., 'Early', 'Mid', 'Late')
    4. moisture (numeric): Soil moisture percentage (0-100)
    5. temperature (numeric): Temperature in Celsius
    6. humidity (numeric): Humidity percentage (0-100)
    7. rainfall (numeric): Rainfall amount in mm
    8. Result (string): Irrigation recommendation ('Irrigate' or 'No Irrigation')
    
    📝 STEPS TO ADD YOUR DATA:
    1. Create a CSV file with the above columns
    2. Fill in your agricultural data
    3. Run: python data_preparation_guide.py
    4. Then run: python train_model_enhanced.py
    
    💡 TIPS:
    - Make sure there are no missing values
    - Use consistent naming for categorical values
    - The more data you have, the better the model will be
    - Try to have balanced data (similar number of 'Irrigate' and 'No Irrigation' cases)
    """)

if __name__ == "__main__":
    print("=== Data Preparation Guide ===\n")
    
    # Show instructions
    prepare_your_data()
    
    # Create sample data
    print("\n" + "="*50)
    print("Creating sample data...")
    create_sample_new_data()
    
    # Validate the sample data
    print("\n" + "="*50)
    print("Validating sample data...")
    validate_data_format('sample_new_data.csv')
