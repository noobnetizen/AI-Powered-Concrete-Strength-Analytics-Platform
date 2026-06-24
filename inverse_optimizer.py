import pickle
import numpy as np
import os

# Set directory path for the trained model files
MODEL_DIR = "models"

def load_forward_models():
    try:
        with open(os.path.join(MODEL_DIR, 'model_7d.pkl'), 'rb') as f:
            model_7d = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'model_28d.pkl'), 'rb') as f:
            model_28d = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
        return model_7d, model_28d, le
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

def get_material_costs():
    return {
        'Cement': 6.0,
        'Ash': 1.5,
        'GGBS': 3.7,
        'Micro': 28.0,
        'Water': 0.1
    }

def predict_strength(model, features):
    return model.predict(np.array([features]))[0]

def optimize_economical_mixture(target_7d, target_28d, model_7d, model_28d, le):
    costs = get_material_costs()
    
    # Baseline mixture [Cement, Type_Enc, Ash, GGBS, Micro, Free_Water, W/B]
    current_mix = np.array([350.0, 0, 50.0, 0.0, 10.0, 157.5, 0.45])
    best_economical_mix = None
    lowest_cost = float('inf')
    
    closest_mix = current_mix.copy()
    min_total_error = float('inf')
    
    iterations = 30000 
    
    print(f"Guided search for most economical mix to hit 7d={target_7d} and 28d={target_28d}...")

    for i in range(iterations):
        trial_mix = current_mix.copy()
        
        # Get current predictions to determine direction of adjustment
        curr_7 = predict_strength(model_7d, current_mix)
        curr_28 = predict_strength(model_28d, current_mix)
        
        # Determine if we need more or less strength
        # We prioritize the 28-day strength for the primary direction
        diff_28 = target_28d - curr_28
        
        # Select a component to adjust
        idx = np.random.choice([0, 2, 3, 4, 6]) 
        
        # DIRECTED TWEAKING: Adjust based on the sign of the error
        if idx == 6: # Water/Binder: Inverse relationship with strength
            adjustment = -0.01 if diff_28 > 0 else 0.01
            trial_mix[idx] = np.clip(trial_mix[idx] + adjustment, 0.20, 0.60)
            trial_mix[5] = trial_mix[0] * trial_mix[6]
        elif idx == 0: # Cement: Direct relationship
            adjustment = 5.0 if diff_28 > 0 else -5.0
            trial_mix[idx] = np.clip(trial_mix[idx] + adjustment, 200, 500)
            trial_mix[5] = trial_mix[0] * trial_mix[6]
        elif idx == 2: # Ash: Direct relationship
            adjustment = 5.0 if diff_28 > 0 else -5.0
            trial_mix[idx] = np.clip(trial_mix[idx] + adjustment, 0, 250)
        elif idx == 3: # GGBS: Direct relationship
            adjustment = 5.0 if diff_28 > 0 else -5.0
            trial_mix[idx] = np.clip(trial_mix[idx] + adjustment, 0, 250)
        elif idx == 4: # Micro: Strong direct relationship
            adjustment = 2.0 if diff_28 > 0 else -2.0
            trial_mix[idx] = np.clip(trial_mix[idx] + adjustment, 0, 100)

        # Predict strengths for the adjusted mix
        pred_7 = predict_strength(model_7d, trial_mix)
        pred_28 = predict_strength(model_28d, trial_mix)
        
        error_7 = abs(target_7d - pred_7)
        error_28 = abs(target_28d - pred_28)
        total_error = error_7 + error_28
        
        # Update the closest mix found overall
        if total_error < min_total_error:
            min_total_error = total_error
            closest_mix = trial_mix.copy()

        # If mixture is within 2 MPa tolerance, check for the cheapest cost
        if error_7 < 2.0 and error_28 < 2.0:
            total_cost = (trial_mix[0] * costs['Cement']) + \
                         (trial_mix[2] * costs['Ash']) + \
                         (trial_mix[3] * costs['GGBS']) + \
                         (trial_mix[4] * costs['Micro']) + \
                         (trial_mix[5] * costs['Water'])
            
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_economical_mix = trial_mix.copy()
                current_mix = trial_mix.copy() 

    final_mix = best_economical_mix if best_economical_mix is not None else closest_mix
    
    if best_economical_mix is None:
        lowest_cost = (final_mix[0] * costs['Cement']) + \
                      (final_mix[2] * costs['Ash']) + \
                      (final_mix[3] * costs['GGBS']) + \
                      (final_mix[4] * costs['Micro']) + \
                      (final_mix[5] * costs['Water'])

    return final_mix, lowest_cost

def run_economic_engineering():
    model_7d, model_28d, le = load_forward_models()
    if model_28d is None: 
        print("Models not found. Please run train_model.py first.")
        return

    try:
        print("\nAI Economical Mix Design System (Directed Search)")
        target_7d = float(input("Enter Target 7-Day Strength (MPa): "))
        target_28d = float(input("Enter Target 28-Day Strength (MPa): "))
        
        result_mix, total_cost = optimize_economical_mixture(target_7d, target_28d, model_7d, model_28d, le)
        
        final_7d = predict_strength(model_7d, result_mix)
        final_28d = predict_strength(model_28d, result_mix)

        print("\nRESULTS SUMMARY")
        print(f"Target Strengths: 7d = {target_7d} | 28d = {target_28d}")
        print(f"Predicted Strengths: 7d = {final_7d:.2f} | 28d = {final_28d:.2f}")
        
        status = "Cheapest Valid Mix" if abs(target_7d-final_7d)<2.0 and abs(target_28d-final_28d)<2.0 else "Closest Available Mix"
        print(f"Estimated Cost: {total_cost:.2f} Units/m3")
        print(f"Status: {status}")
        
        print("\nRECOMMENDED MIXTURE DESIGN")
        print(f"Cement: {result_mix[0]:.2f} kg/m3")
        print(f"Type of Cement: {le.inverse_transform([int(result_mix[1])])[0]}")
        print(f"Ash: {result_mix[2]:.2f} kg/m3")
        print(f"GGBS: {result_mix[3]:.2f} kg/m3")
        print(f"Micro-silica: {result_mix[4]:.2f} kg/m3")
        print(f"Free Water: {result_mix[5]:.2f} kg/m3")
        print(f"Water/Binder: {result_mix[6]:.4f}")
    except ValueError:
        print("Please enter valid numbers.")
if __name__ == "__main__":
    run_economic_engineering()