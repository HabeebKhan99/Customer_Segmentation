# Libraries imports
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Title Organizing
st.set_page_config(page_title = "Customer Segmentation", page_icon = "🛍️", layout = "centered")

# Premium UI

st.markdown("""
<style>

/* App Background */
html, body, .stApp{
    background-color:#f5f7fa !important;
}

[data-testid="stAppViewcontainer"]
{
    background-color:#f5f7fa !important;
}

/* Main Title */
h1{
    text-align:center;
    color:#1f2937;
    font-weight:bold
}

/* Sub Heading */
h2{
    color:#374151;
    font-weight:bold;
}

/* Predict Button */
.stButton > button{
    background-color:#2563eb !important;
    color:white !important;
    border:none !important;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
    font-weight:bold;
    
}

/* Button Hover */

.stButton > button:hover{
    background-color:#1d4ed8 !important;
    color:white !important;
}

/* Input Boxes */
.stNumberInput input,
.stSelectbox
div[data-baseweb = "select"]
{
    border-radius:8px;
}

/* Prediction Box */
.stSuccess{
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html = True)

# BASE_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths

model_path = os.path.join(BASE_DIR, "models/model.pkl")
scaler_path = os.path.join(BASE_DIR, "models/scaler.pkl")
feature_path = os.path.join(BASE_DIR, "models/feature.pkl")
encoder_path = os.path.join(BASE_DIR, "models/encoder.pkl")

print(__file__)
print(os.getcwd())
print(BASE_DIR)

# # Load Trained Model Files

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature = joblib.load(feature_path)
encoder = joblib.load(encoder_path)

# Main Title

st.title("🛍️ Customer Segmentation System")

# Description
st.write("### Fill In The Customer Information Below To Identify The Predicted Customer Group.")

# User Inputs

Gender = st.selectbox("Gender",["M","F"])
Age = st.number_input("Age",min_value=1,max_value=100,value=25)
Income = st.number_input("Annual Income (k$)", min_value=1,value=50)
Score = st.number_input("Spending Score (1-100)",min_value=1,max_value=100,value=50)

# Prediction Button
Predict = st.button("Predict Customer Group")

 # convert Input To DataFrame

if Predict:
    input_data = pd.DataFrame({
        "Gender":[Gender],
        "Age":[Age],
        "Annual Income (k$)":[Income],
        "Spending Score (1-100)":[Score]
    })

     # Encoding Categorical Features

    input_data["Gender"] = encoder.transform(input_data["Gender"])

     # Align Features With Training Data
    input_data = input_data.reindex(columns=feature,fill_value=0)

     # Scaling 
    input_data = scaler.transform(input_data)

    # Final Prediction

    Prediction = model.predict(input_data)

    if Prediction[0] == 0:
        st.success("Customer Belongs To Group 1")
    elif Prediction[0] == 1:
        st.success("Customer Belongs To Group 2")
    elif Prediction[0] == 2:
        st.success("Customer Belongs To Group 3")
    elif Prediction [0] == 3:
        st.success("Customer Belongs To Group 4")


# Footer

st.markdown(""" <div style = "text-align:center;
color:gray;">
        
Built with ❤️ using <b>Python</b>
,<b>Streamlit</b>
,<b>Numpy</b>
,<b>Pandas</b> and 
<b> Scikit-learn</b>
        
<br><br>

Developed by <b> HABEEB KHAN </b>

</div>
""" , unsafe_allow_html = True)
      
