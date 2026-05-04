import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# --- 2. HIDE GITHUB ICON & STREAMLIT MENU ---
# This CSS hides the top header (GitHub/Deploy) and the hamburger menu
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            .stAppDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
try:
    model = joblib.load('iris_model.pkl')
    le = joblib.load('label_encoder.pkl')
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# --- 4. HEADER ---
st.title("🌸 Iris Flower Classifier")
st.write("Enter the measurements below to identify the species.")

# --- 5. SIDEBAR / INPUTS ---
st.sidebar.header("Input Measurements")
sl = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sw = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
pl = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
pw = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# --- 6. PREDICTION LOGIC ---
if st.button("Predict Species"):
    if model_loaded:
        # Create the dataframe for the model
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
        st.error("⚠️ Error: Model files not found. Ensure .pkl files are in the repository.")
