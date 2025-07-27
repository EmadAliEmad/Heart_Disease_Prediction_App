import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# Define the path to the model file
# This path is relative to the project root, assuming streamlit_app.py is in 'ui'
MODEL_PATH = '../models/best_svm_heart_disease_model.joblib'

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    st.success(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}")
    st.error("Please ensure the model is saved in the 'models' directory.")
    model = None
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    model = None

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("""
This application predicts the likelihood of heart disease based on various health parameters.
Please enter the patient's information below.
""")

# --- Input Features (based on X_selected from 03_feature_selection.ipynb) ---
# The order of features must match the training data:
# ['thalach', 'oldpeak', 'thal_7.0', 'cp_4', 'age', 'chol', 'trestbps',
#  'exang_1', 'slope_2', 'sex_1', 'ca_1.0', 'cp_3']

st.header("Patient Health Parameters")

# Create input fields for each feature
# Using default values as a guide, these are NOT scaled values
# Note: For actual deployment, you might want to reverse-scale these or
# provide guidance on what raw values to enter for better user experience.
# For now, we assume user enters *preprocessed/scaled-like* values as model expects.
# (The model expects values like the ones after preprocessing and feature selection)

# It's crucial to map input fields to the *exact* 12 features chosen and their order.
# For simplicity, we'll ask for raw values and indicate scaling/encoding is handled conceptually.
# In a real app, you'd add preprocessing steps *before* passing to the model.
# For this example, we'll just gather the 12 features in order.

# Let's map to original meaningful names for user input, then internally use selected features.
# To properly handle the UI, we should ask for original features and then apply the same
# preprocessing (scaling, one-hot encoding) that was applied during training.
# However, for simplicity given the current scope, we will ask for the 12 selected features directly
# as if they are already preprocessed. In a full production app, you would include the scaler
# and encoder objects as part of the model pipeline or load them separately.

# For now, let's create placeholders for input fields mirroring the selected features.
# A more robust solution would involve loading the scaler/encoder used in data_preprocessing.ipynb
# and feature_selection.ipynb and applying them to raw user inputs.

st.subheader("Numerical Features (Scaled or Processed-like values for demonstration):")
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", value=0.0, format="%.3f", help="Scaled value representing thalach.")
oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak)", value=0.0, format="%.3f", help="Scaled value representing oldpeak.")
age = st.number_input("Age", value=0.0, format="%.3f", help="Scaled value representing age.")
chol = st.number_input("Serum Cholestoral in mg/dl (chol)", value=0.0, format="%.3f", help="Scaled value representing chol.")
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=0.0, format="%.3f", help="Scaled value representing trestbps.")

st.subheader("Categorical Features (Encoded values - 0 or 1):")
thal_7_0 = st.selectbox("Thal: Normal (0), Fixed Defect (1 if 7.0 was selected in original)", options=[0, 1], index=0, help="Is thal '7.0'? (1 for yes, 0 for no). Original thal: 3.0 = normal; 6.0 = fixed defect; 7.0 = reversible defect.")
cp_4 = st.selectbox("Chest Pain Type: Asymptomatic (1 if cp_4 was selected in original)", options=[0, 1], index=0, help="Is chest pain type '4' (asymptomatic)? (1 for yes, 0 for no). Original cp: 1=typical angina, 2=atypical angina, 3=non-anginal pain, 4=asymptomatic.")
exang_1 = st.selectbox("Exercise Induced Angina: Yes (1) / No (0)", options=[0, 1], index=0, help="Exercise induced angina? (1 for yes, 0 for no).")
slope_2 = st.selectbox("Slope of the peak exercise ST segment: Upsloping (0) / Flat (1 if slope_2 was selected in original) / Downsloping", options=[0, 1], index=0, help="Is the slope '2' (flat)? (1 for yes, 0 for no). Original slope: 1=upsloping, 2=flat, 3=downsloping.")
sex_1 = st.selectbox("Sex: Male (1) / Female (0)", options=[0, 1], index=0, help="Is sex '1' (male)? (1 for yes, 0 for no).")
ca_1_0 = st.selectbox("Number of major vessels (0-3) colored by flourosopy: 1 (1 if ca_1.0 was selected in original)", options=[0, 1], index=0, help="Is 'ca' '1.0'? (1 for yes, 0 for no). Original ca: 0-3.")
cp_3 = st.selectbox("Chest Pain Type: Non-anginal Pain (1 if cp_3 was selected in original)", options=[0, 1], index=0, help="Is chest pain type '3' (non-anginal pain)? (1 for yes, 0 for no).")


# Order of features as per X_selected_features.csv (from feature selection)
# ['thalach', 'oldpeak', 'thal_7.0', 'cp_4', 'age', 'chol', 'trestbps',
#  'exang_1', 'slope_2', 'sex_1', 'ca_1.0', 'cp_3']
features = [
    thalach, oldpeak, thal_7_0, cp_4, age, chol, trestbps,
    exang_1, slope_2, sex_1, ca_1_0, cp_3
]

# Prediction button
if st.button("Predict Heart Disease"):
    if model is not None:
        try:
            # Convert features to numpy array and reshape for prediction
            input_data = np.array(features).reshape(1, -1)

            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            predicted_class = int(prediction[0])
            probability_no_disease = float(prediction_proba[0][0])
            probability_disease = float(prediction_proba[0][1])

            st.subheader("Prediction Result:")
            if predicted_class == 1:
                st.error(f"Prediction: Disease Present (Probability: {probability_disease:.2%})")
                st.write(f"Based on the input parameters, the model predicts a **{probability_disease:.2%}** chance of heart disease.")
            else:
                st.success(f"Prediction: No Disease (Probability: {probability_no_disease:.2%})")
                st.write(f"Based on the input parameters, the model predicts a **{probability_no_disease:.2%}** chance of no heart disease.")

            st.info("Note: These predictions are based on the trained machine learning model and should not be used as medical advice.")

            # Optional: Display feature importances or other insights (for future enhancement)
            # st.subheader("Feature Contribution (Example - Needs specific implementation):")
            # If you had SHAP or LIME, you could visualize local explanations here.

        except ValueError as e:
            st.error(f"Input Error: {e}. Please ensure all fields are filled correctly.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
    else:
        st.warning("Model not loaded. Cannot make predictions.")