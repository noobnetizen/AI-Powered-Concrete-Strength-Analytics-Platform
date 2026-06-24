import pickle
import pandas as pd
import numpy as np
import os

# Set directory path for the trained model files
MODEL_DIR = "models"

def get_prediction():
    try:
        # Load the saved models, encoder, and performance metrics
        with open(os.path.join(MODEL_DIR, 'model_7d.pkl'), 'rb') as f:
            model_7d = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'model_28d.pkl'), 'rb') as f:
            model_28d = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
        
        print("\nConcrete Strength Prediction System")
        print("Enter the values for the red fields:\n")

        # Collect material inputs from user
        cement = float(input("Cement (kg/m3): "))
        type_of_cement = input("Type of Cement (e.g., OPC, PPC): ")
        ash = float(input("Ash (kg/m3): "))
        ggbs = float(input("GGBS (kg/m3): "))
        micro = float(input("Micro-silica (kg/m3): "))
        free_water = float(input("Free Water (kg/m3): "))
        water_binder = float(input("Water/Binder ratio: "))

        # Encode categorical cement type using the saved LabelEncoder
        try:
            type_encoded = le.transform([type_of_cement])[0]
        except ValueError:
            print(f"Warning: {type_of_cement} was not in training data. Using default value.")
            type_encoded = 0

        # Format inputs into a feature array for the XGBoost model
        features = np.array([[cement, type_encoded, ash, ggbs, micro, free_water, water_binder]])

        # Generate strength predictions for both 7 and 28 days
        pred_7 = model_7d.predict(features)[0]
        pred_28 = model_28d.predict(features)[0]

        # Display the predicted strength results
        print("\nPREDICTION RESULTS")
        print(f"Predicted 7-Day Strength:  {pred_7:.2f} MPa")
        print(f"Predicted 28-Day Strength: {pred_28:.2f} MPa")
        
        # Display the overall model accuracy metrics from the training phase
        print("\nMODEL PERFORMANCE METRICS")
        print(f"7-Day Model:  R2 = {metrics['7d']['R2']:.4f}, MAE = {metrics['7d']['MAE']:.2f}, RMSE = {metrics['7d']['RMSE']:.2f}")
        print(f"28-Day Model: R2 = {metrics['28d']['R2']:.4f}, MAE = {metrics['28d']['MAE']:.2f}, RMSE = {metrics['28d']['RMSE']:.2f}")

    except FileNotFoundError:
        print("Error: Model files not found! Please run train_model.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    get_prediction()