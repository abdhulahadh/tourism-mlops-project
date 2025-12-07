
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Config
HF_USERNAME = "abdhulahadh"
MODEL_REPO = f"{HF_USERNAME}/tourism-model"
MODEL_FILENAME = "model.joblib"

st.set_page_config(page_title="GL - Tourism Package Prediction", layout="wide")

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

st.title("GL-UT Austin Tourism Package Purchase Prediction")
st.write("Enter customer details below to predict if they will purchase the Wellness Tourism Package.")

# Input Form
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)
        monthly_income = st.number_input("Monthly Income", min_value=1000, value=20000)
        pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])

    with c3:
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        num_person = st.number_input("Number of Person Visiting", 1, 10, 2)
        num_trips = st.number_input("Number of Trips", 0, 50, 2)
        num_children = st.number_input("Number of Children", 0, 5, 0)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    with col_b:
        own_car = st.selectbox("Owns Car?", ["Yes", "No"])
    with col_c:
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
    
    # Hidden numerical inputs (defaults)
    num_followups = 4.0 # Default/Median

    submit_button = st.form_submit_button("Predict")

if submit_button and model:
    # Prepare Input Data
    # Mapping Yes/No to 1/0
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [num_person],
        'NumberOfFollowups': [num_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips],
        'Passport': [1 if passport == "Yes" else 0],
        'PitchSatisfactionScore': [pitch_score],
        'OwnCar': [1 if own_car == "Yes" else 0],
        'NumberOfChildrenVisiting': [num_children],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    })
    
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    st.divider()
    if prediction == 1:
        st.success(f" Prediction: **Likely to Purchase** (Probability: {prob:.2f})")
    else:
        st.warning(f" Prediction: **Unlikely to Purchase** (Probability: {prob:.2f})")
