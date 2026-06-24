import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import shap

app = Flask(__name__, template_folder='templates')

# Get the absolute path to the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

try:
    with open(os.path.join(MODEL_DIR, 'model_7d.pkl'), 'rb') as f:
        model_7d = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_28d.pkl'), 'rb') as f:
        model_28d = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'metrics.pkl'), 'rb') as f:
        metrics = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'shap_importance.pkl'), 'rb') as f:
        global_importance = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    model_7d = model_28d = le = metrics = global_importance = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract values
        cement = float(data.get('cement', 0))
        type_of_cement = data.get('type_of_cement', 'OPC')
        ash = float(data.get('ash', 0))
        ggbs = float(data.get('ggbs', 0))
        micro = float(data.get('micro', 0))
        free_water = float(data.get('free_water', 0))
        water_binder = float(data.get('water_binder', 0))
        
        # Encode categorical
        type_encoded = le.transform([type_of_cement])[0]
        
        # Create feature array
        features = np.array([[cement, type_encoded, ash, ggbs, micro, free_water, water_binder]])
        
        # Predictions
        pred_7d = model_7d.predict(features)[0]
        pred_28d = model_28d.predict(features)[0]
        
        # Local SHAP explanation for 28d
        explainer = shap.TreeExplainer(model_28d)
        shap_vals = explainer.shap_values(features)
        
        feature_names = ['Cement', 'Type of Cement', 'Ash', 'GGBS', 'Micro', 'Free Water', 'Water/Binder']
        local_explanation = []
        for i in range(len(feature_names)):
            local_explanation.append({
                'feature': feature_names[i],
                'impact': float(shap_vals[0][i])
            })
        
        # Classification
        strength_class = "Normal"
        if pred_28d > 60:
            strength_class = "High Strength Concrete"
        if pred_28d > 80:
            strength_class = "Very High Strength Concrete"
        if pred_28d < 30:
            strength_class = "Low Strength Concrete"

        return jsonify({
            'strength_7d': round(float(pred_7d), 2),
            'strength_28d': round(float(pred_28d), 2),
            'classification': strength_class,
            'local_shap': local_explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

@app.route('/shap_importance', methods=['GET'])
def get_shap():
    return jsonify(global_importance.to_dict(orient='records'))

@app.route('/dataset_info', methods=['GET'])
def get_info():
    # Return some synthetic dataset info
    return jsonify({
        'samples': 1030,
        'features': 7,
        'r2_avg': (metrics['7d']['R2'] + metrics['28d']['R2']) / 2,
        'rmse_avg': (metrics['7d']['RMSE'] + metrics['28d']['RMSE']) / 2,
        'pred_time': metrics['28d']['PredTime']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
