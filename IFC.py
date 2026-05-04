import streamlit as st
import pandas as pd
import joblib

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Iris Species Predictor",
    layout="centered"
)

# --- 2. HIDE STREAMLIT / GITHUB UI ELEMENTS ---
hide_st_style = """
<style>

/* Hide Streamlit menu */
#MainMenu {
    visibility: hidden;
}

/* Hide header */
header {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Hide deploy button */
.stAppDeployButton {
    display: none;
}

/* Hide toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide status widget */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Hide connection status */
#stConnectionStatus {
    display: none !important;
}

/* Hide bottom-right decoration */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide floating GitHub/creator badge */
.viewerBadge_container__1QSob {
    display: none !important;
}

/* Optional: cleaner padding */
.block-container {
    padding-top: 2rem;
}

</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 3. LOAD MODEL FILES ---
try:
    model = joblib.load("iris_model.pkl")
    le = joblib.load("label_encoder.pkl")
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# --- 4. MAIN HEADER ---
st.title("🌸 Iris Flower Classifier")

st.write(
    "Enter the flower measurements below to predict the Iris species."
)

# --- 5. SIDEBAR INPUTS ---
st.sidebar.header("Input Measurements")

sl = st.sidebar.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sw = st.sidebar.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

pl = st.sidebar.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

pw = st.sidebar.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# --- 6. PREDICTION SECTION ---
if st.button("Predict Species"):

    if model_loaded:

        # Create dataframe for prediction
        input_data = pd.DataFrame(
            [[sl, sw, pl, pw]],
            columns=[
                'sepal_length',
                'sepal_width',
                'petal_length',
                'petal_width'
            ]
        )

        # Predict species
        pred_encoded = model.predict(input_data)

        # Decode label
        species = le.inverse_transform(pred_encoded)[0]

        # Display prediction
        st.success(f"Predicted Species: {species}")

        # Display corresponding flower image
        if species == 'Iris-setosa':

            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                caption="Iris Setosa",
                width=400
            )

        elif species == 'Iris-versicolor':

            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                caption="Iris Versicolor",
                width=400
            )

        else:

            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
                caption="Iris Virginica",
                width=400
            )

    else:

        st.error(
            "⚠️ Error: Model files not found. "
            "Make sure the .pkl files exist in the repository."
        )
