from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load components
try:
    clf = joblib.load('fertilizer_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le_soil = joblib.load('le_soil.joblib')
    le_crop = joblib.load('le_crop.joblib')
    le_fertilizer = joblib.load('le_fertilizer.joblib')
except Exception as e:
    print(f"Error loading model components: {e}")

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

@app.route('/')
def index():
    return render_template('index.html', 
                          soil_types=le_soil.classes_.tolist(),
                          crop_types=le_crop.classes_.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        soil_type = data['soil_type']
        crop_type = data['crop_type']
        n = float(data['nitrogen'])
        k = float(data['potassium'])
        p = float(data['phosphorous'])
        
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
        
        # Calculate optimal values from dataset for this crop type
        df = pd.read_csv('Fertilizer Prediction.csv')
        df.columns = df.columns.str.strip()
        optimal_data = df[df['Crop Type'] == crop_type]
        
        optimal_values = {
            'N': float(optimal_data['Nitrogen'].mean()),
            'K': float(optimal_data['Potassium'].mean()),
            'P': float(optimal_data['Phosphorous'].mean())
        }
        
        return jsonify({
            'recommendation': fertilizer_name,
            'code': fertilizer_code,
            'current_values': {'N': n, 'K': k, 'P': p},
            'optimal_values': optimal_values
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
