import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

#load dataset
df = pd.read_csv("crop_weather_dataset.csv")

#encoding categorical
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_stage = LabelEncoder()

df['crop'] = le_crop.fit_transform(df['crop'])
df['soil_type'] = le_soil.fit_transform(df['soil_type'])
df['Seedling Stage'] = le_stage.fit_transform(df['Seedling Stage'])


X = df.drop('Result', axis=1)
y = df['Result']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train, y_train)

#accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")



#.pkl file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Trained model saved as 'decision_tree_model.pkl'")
