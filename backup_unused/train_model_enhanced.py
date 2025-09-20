import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_model_with_data(dataset_path="Irrigationdatset.csv", model_name="decision_tree_model.pkl"):
    """
    Train a decision tree model with the specified dataset
    
    Args:
        dataset_path (str): Path to the CSV dataset
        model_name (str): Name for the saved model file
    """
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found!")
        return None
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values if any
    if df.isnull().sum().any():
        print("Warning: Missing values found. Filling with mode for categorical and mean for numerical columns.")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    
    # Encoding categorical variables
    print("\nEncoding categorical variables...")
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_stage = LabelEncoder()
    
    # Check which columns exist and encode accordingly
    categorical_columns = []
    if 'crop' in df.columns:
        df['crop'] = le_crop.fit_transform(df['crop'])
        categorical_columns.append('crop')
    if 'soil_type' in df.columns:
        df['soil_type'] = le_soil.fit_transform(df['soil_type'])
        categorical_columns.append('soil_type')
    if 'Seedling Stage' in df.columns:
        df['Seedling Stage'] = le_stage.fit_transform(df['Seedling Stage'])
        categorical_columns.append('Seedling Stage')
    
    print(f"Encoded categorical columns: {categorical_columns}")
    
    # Prepare features and target
    if 'Result' not in df.columns:
        print("Error: 'Result' column not found in dataset!")
        return None
    
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Decision Tree Classifier
    print("\nTraining Decision Tree Classifier...")
    model = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Detailed performance metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"\nTrained model saved as '{model_name}'")
    
    # Save label encoders for future use
    encoders = {
        'crop_encoder': le_crop if 'crop' in categorical_columns else None,
        'soil_encoder': le_soil if 'soil_type' in categorical_columns else None,
        'stage_encoder': le_stage if 'Seedling Stage' in categorical_columns else None
    }
    
    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)
    
    print("Label encoders saved as 'label_encoders.pkl'")
    
    return model, accuracy

def train_with_multiple_datasets(dataset_paths, model_name="decision_tree_model.pkl"):
    """
    Train model with multiple datasets combined
    
    Args:
        dataset_paths (list): List of paths to CSV datasets
        model_name (str): Name for the saved model file
    """
    print("Combining multiple datasets...")
    
    combined_df = pd.DataFrame()
    
    for i, path in enumerate(dataset_paths):
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded dataset {i+1}: {path} (Shape: {df.shape})")
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Warning: Dataset {path} not found, skipping...")
    
    if combined_df.empty:
        print("Error: No valid datasets found!")
        return None
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Save combined dataset
    combined_df.to_csv("combined_dataset.csv", index=False)
    print("Combined dataset saved as 'combined_dataset.csv'")
    
    # Train with combined data
    return train_model_with_data("combined_dataset.csv", model_name)

if __name__ == "__main__":
    print("=== Agricultural Irrigation Model Training ===\n")
    
    # Option 1: Train with existing dataset
    print("Training with existing dataset...")
    model, accuracy = train_model_with_data()
    
    if model is not None:
        print(f"\n✅ Training completed successfully!")
        print(f"Model accuracy: {accuracy:.4f}")
    else:
        print("\n❌ Training failed!")
    
    # Option 2: Train with multiple datasets (uncomment to use)
    # print("\n" + "="*50)
    # print("Training with multiple datasets...")
    # dataset_paths = ["Irrigationdatset.csv", "new_dataset.csv", "another_dataset.csv"]
    # model, accuracy = train_with_multiple_datasets(dataset_paths)
