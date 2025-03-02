import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, scaler, and columns list
model = joblib.load('classification_model_personal.pkl')
scaler = joblib.load('c_scaler.pkl')
columns = joblib.load('c_X_train.pkl')

# Function to preprocess the input data (like scaling and one-hot encoding)
def preprocess_input(user_input):
    # Apply necessary preprocessing steps here (scaling, encoding, etc.)
    # Example: Convert input into a DataFrame and apply transformations
    input_data = pd.DataFrame([user_input], columns=columns)
    
    # Apply scaling
    input_data[['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']] = scaler.transform(input_data[['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']])
    
    # Apply one-hot encoding
    input_data = pd.get_dummies(input_data, columns=['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX', 'LNPAYFREQ', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED', 'LNPERIOD_CATEGORY'], drop_first=True)
    
    # Ensure the input matches the columns used during training
    input_data = input_data.reindex(columns=columns, fill_value=0)
    
    return input_data

# Title of the app
st.title("Loan Default Risk Prediction")

# Input fields for user data
LNAMOUNT = st.number_input("Loan Amount", min_value=0)
LNINTRATE = st.number_input("Interest Rate", min_value=0.0)
LNINSTAMT = st.number_input("Installment Amount", min_value=0)
AGE = st.number_input("Age", min_value=18)
QSPURPOSEDES = st.selectbox("Loan Purpose", ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
QS_SECTOR = st.selectbox("Sector", ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])
LNBASELDESC = st.selectbox("Loan Base Description", ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
SEX = st.selectbox("Gender", ['M', 'F'])
LNPAYFREQ = st.selectbox("Payment Frequency", [,'2','5', '12'])
CREDIT_CARD_USED = st.selectbox("Credit Card Used", ['Yes', 'No'])
DEBIT_CARD_USED = st.selectbox("Debit Card Used", ['Yes', 'No'])
LNPERIOD_CATEGORY = st.selectbox("Loan Period Category", ['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM'])

# Collect user inputs in a dictionary
user_input = {
    'LNAMOUNT': LNAMOUNT,
    'LNINTRATE': LNINTRATE,
    'LNINSTAMT': LNINSTAMT,
    'AGE': AGE,
    'QSPURPOSEDES': QSPURPOSEDES,
    'QS_SECTOR': QS_SECTOR,
    'LNBASELDESC': LNBASELDESC,
    'SEX': SEX,
    'LNPAYFREQ': LNPAYFREQ,
    'CREDIT_CARD_USED': CREDIT_CARD_USED,
    'DEBIT_CARD_USED': DEBIT_CARD_USED,
    'LNPERIOD_CATEGORY': LNPERIOD_CATEGORY
}

# Preprocess the input data
input_data = preprocess_input(user_input)

# Make prediction
prediction = model.predict(input_data)

# Show result
if prediction == 1:
    st.success("Loan Default Risk: Default")
else:
    st.success("Loan Default Risk: No Default")
