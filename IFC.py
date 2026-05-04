import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# --- LOAD MODELS ---
# We load these at the top so they are ready before the user clicks anything
try:
    model = joblib.load('iris_model.pkl')
    le = joblib.load('label_encoder.pkl')
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# --- HEADER ---
st.title("🌸 Iris Flower Classifier")
st.write("Enter the measurements below to identify the species.")

# --- SIDEBAR / INPUTS ---
st.sidebar.header("Input Measurements")
sl = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sw = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
pl = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
pw = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# --- PREDICTION LOGIC ---
if st.button("Predict Species"):
    if model_loaded:
        # Create the dataframe for the model (matching your training columns)
        input_data = pd.DataFrame([[sl, sw, pl, pw]], 
                                 columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        # Make Prediction
        pred_encoded = model.predict(input_data)
        species = le.inverse_transform(pred_encoded)[0]
        
        # --- OUTPUT ---
        st.success(f"### Result: {species}")
        
        # Display the corresponding image
        if species == 'Iris-setosa':
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa", width=400)
        elif species == 'Iris-versicolor':
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor", width=400)
        else:
            st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica", width=400)
    else:
        st.error("⚠️ Error: 'iris_model.pkl' or 'label_encoder.pkl' not found in this folder. Please run the save code in your notebook first!")

