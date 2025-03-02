import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load saved models and scaler
classification_model = joblib.load('classification_model_personal.pkl')
c_scaler = joblib.load('c_scaler.pkl')
c_X_train = joblib.load('c_X_train.pkl')

# Title
st.title('Loan Default Risk Predictor')

# User input fields
LNAMOUNT = st.number_input("Loan Amount", min_value=1000, step=1000, format="%.2f")
LNINTRATE = st.number_input("Interest Rate (%)", min_value=0.1, step=0.1, format="%.2f")
LNPERIOD = st.slider("Loan Period (Months)", min_value=6, max_value=360, step=6)
LNINSTAMT = st.number_input("Installment Amount", min_value=100, step=100, format="%.2f")
LNPAYFREQ = st.selectbox("Payment Frequency", [1, 2, 3, 4])
QSPURPOSEDES = st.selectbox("Purpose of Loan", ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
LNBASELDESC = st.selectbox("Loan Type", ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
SEX = st.selectbox("Gender", ["M", "F"])
AGE = st.slider("Age", min_value=18, max_value=80, step=1)
CREDIT_CARD_USED = st.selectbox("Credit Card Used", ["Yes", "No"])
DEBIT_CARD_USED = st.selectbox("Debit Card Used", ["Yes", "No"])
QS_SECTOR = st.selectbox("Sector", ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])
AVERAGE_SAGBAL = st.number_input("Average Savings Account Balance", min_value=0.0, step=100.0, format="%.2f")

# Prediction button
if st.button("Predict Default Risk"):
    # Prepare input data
    input_data = {
        'LNAMOUNT': LNAMOUNT,
        'LNINTRATE': LNINTRATE,
        'LNPERIOD': LNPERIOD,
        'LNINSTAMT': LNINSTAMT,
        'LNPAYFREQ': LNPAYFREQ,
        'AGE': AGE,
        'CREDIT_CARD_USED': 1 if CREDIT_CARD_USED == "Yes" else 0,
        'DEBIT_CARD_USED': 1 if DEBIT_CARD_USED == "Yes" else 0,
        'AVERAGE_SAGBAL': AVERAGE_SAGBAL,
        'QSPURPOSEDES_' + QSPURPOSEDES: 1,
        'LNBASELDESC_' + LNBASELDESC: 1,
        'SEX_' + SEX: 1,
        'QS_SECTOR_' + QS_SECTOR: 1
    }
    
    # Create a DataFrame and align with training features
    input_df = pd.DataFrame([input_data])
    for col in c_X_train:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[c_X_train.columns]
    
    # Scale numerical features
    numeric_features = ['LNAMOUNT', 'LNINTRATE', 'LNPERIOD', 'LNINSTAMT', 'LNPAYFREQ', 'AGE', 'AVERAGE_SAGBAL']
    input_df[numeric_features] = c_scaler.transform(input_df[numeric_features])
    
    # Predict default risk
    prediction = classification_model.predict(input_df)
    result = "Default" if prediction[0] == 1 else "No Default"
    
    # Display prediction result
    st.write(f"Prediction: {result}")

# Reset button
if st.button("Reset"):
    st.experimental_rerun()
