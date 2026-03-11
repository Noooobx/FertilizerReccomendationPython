import joblib
import pandas as pd
import numpy as np
import sys

def predict_fertilizer():
    # Load components
    try:
        clf = joblib.load('fertilizer_model.joblib')
        scaler = joblib.load('scaler.joblib')
        le_soil = joblib.load('le_soil.joblib')
        le_crop = joblib.load('le_crop.joblib')
        le_fertilizer = joblib.load('le_fertilizer.joblib')
    except FileNotFoundError:
        print("Error: Model or encoders not found. Please run train_model.py first.")
        return

    # Descriptive mapping for fertilizers
    FERTILIZER_DESCRIPTIONS = {
        'Urea': 'Urea (High Nitrogen content)',
        'DAP': 'DAP (Diammonium Phosphate - Rich in Phosphorous)',
        '14-35-14': 'Compound Fertilizer NPK 14-35-14 (Phosphorous Rich)',
        '28-28': 'Ammonium Phosphate Sulphate (28-28-0)',
        '17-17-17': 'Balanced Complex Fertilizer NPK 17-17-17',
        '20-20': 'Ammonium Phosphate Sulphate (20-20-0)',
        '10-26-26': 'Potash-Rich Complex Fertilizer NPK 10-26-26'
    }

    print("--- Fertilizer Recommendation System ---")
    
    # Get inputs
    try:
        temp = float(input("Enter Temperature: "))
        humidity = float(input("Enter Humidity: "))
        moisture = float(input("Enter Moisture: "))
        
        print(f"Available Soil Types: {le_soil.classes_.tolist()}")
        soil_type = input("Enter Soil Type: ")
        
        print(f"Available Crop Types: {le_crop.classes_.tolist()}")
        crop_type = input("Enter Crop Type: ")
        
        n = float(input("Enter Nitrogen (N): "))
        k = float(input("Enter Potassium (K): "))
        p = float(input("Enter Phosphorous (P): "))
        
        # Prepare data
        input_data = pd.DataFrame([[temp, humidity, moisture, soil_type, crop_type, n, k, p]], 
                                 columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
        
        # Transform categorical inputs
        input_data['Soil Type'] = le_soil.transform([soil_type])[0]
        input_data['Crop Type'] = le_crop.transform([crop_type])[0]
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = clf.predict(input_scaled)
        fertilizer_code = le_fertilizer.inverse_transform(prediction)[0]
        
        # Get textual representation
        fertilizer_name = FERTILIZER_DESCRIPTIONS.get(fertilizer_code, fertilizer_code)
        
        print(f"\nRecommended Fertilizer: {fertilizer_name}")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict_fertilizer()
