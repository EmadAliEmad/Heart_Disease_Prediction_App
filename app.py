# app.py

import joblib
from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

# Define the path to the model file
# This path is relative to where app.py is located
MODEL_PATH = 'models/best_svm_heart_disease_model.joblib'

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure the model is saved in the 'models' directory.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

@app.route('/')
def home():
    return "Heart Disease Prediction API. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        # Get data from the request
        data = request.get_json(force=True)

        # Ensure the input data has 12 features as expected by the model
        # The order of features must match the training data:
        # ['thalach', 'oldpeak', 'thal_7.0', 'cp_4', 'age', 'chol', 'trestbps',
        #  'exang_1', 'slope_2', 'sex_1', 'ca_1.0', 'cp_3']

        # Example expected input:
        # {
        #     "features": [
        #         0.017197,  # thalach
        #         1.087338,  # oldpeak
        #         0,         # thal_7.0
        #         0,         # cp_4
        #         0.948726,  # age
        #         -0.264900, # chol
        #         0.757525,  # trestbps
        #         0,         # exang_1
        #         0,         # slope_2
        #         1,         # sex_1
        #         0,         # ca_1.0
        #         0          # cp_3
        #     ]
        # }

        features = data['features']

        # Convert to numpy array and reshape for single prediction
        # The model expects a 2D array (n_samples, n_features)
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        prediction_proba = model.predict_proba(features_array)

        # Get the predicted class (0 or 1)
        predicted_class = int(prediction[0])

        # Get the probabilities for each class
        probability_no_disease = float(prediction_proba[0][0])
        probability_disease = float(prediction_proba[0][1])

        # Map the predicted class to a readable label
        result_label = "Disease Present" if predicted_class == 1 else "No Disease"

        return jsonify({
            'prediction': predicted_class,
            'prediction_label': result_label,
            'probability_no_disease': probability_no_disease,
            'probability_disease': probability_disease
        })

    except KeyError as e:
        return jsonify({'error': f"Missing key in JSON input: {e}. Please provide 'features' key with 12 values."}), 400
    except ValueError as e:
        return jsonify({'error': f"Invalid input format or number of features: {e}. Expected 12 features."}), 400
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    # To run the app:
    # 1. Open your terminal.
    # 2. Navigate to the Heart_Disease_Project directory (where app.py is).
    # 3. Make sure your virtual environment is activated.
    # 4. Run: python app.py
    # The API will be available at http://127.0.0.1:5000/
    # You can test it using tools like Postman or curl.
    # Example curl command for testing (replace with actual feature values):
    # curl -X POST -H "Content-Type: application/json" \
    # -d '{"features": [0.017, 1.087, 0, 0, 0.948, -0.264, 0.757, 0, 0, 1, 0, 0]}' \
    # http://127.0.0.1:5000/predict
    app.run(debug=True) # debug=True reloads the server on code changes