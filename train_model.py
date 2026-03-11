import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model():
    # Load dataset
    file_path = '/home/nandakishor/works/FIX/Fertilizer/Fertilizer Prediction.csv'
    df = pd.read_csv(file_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    print("Columns:", df.columns.tolist())
    
    # Encode categorical features
    le_soil = LabelEncoder()
    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
    
    le_crop = LabelEncoder()
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
    
    le_fertilizer = LabelEncoder()
    df['Fertilizer Name'] = le_fertilizer.fit_transform(df['Fertilizer Name'])
    
    # Features and Target
    X = df.drop('Fertilizer Name', axis=1)
    y = df['Fertilizer Name']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_fertilizer.classes_))
    
    # Save Everything
    joblib.dump(clf, 'fertilizer_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le_soil, 'le_soil.joblib')
    joblib.dump(le_crop, 'le_crop.joblib')
    joblib.dump(le_fertilizer, 'le_fertilizer.joblib')
    
    print("\nModel and encoders saved successfully.")

if __name__ == "__main__":
    train_model()
