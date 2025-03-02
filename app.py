import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open("classification_model_personal.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("c_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load reference training data structure
X_train_ref = pickle.load(open("c_X_train.pkl", "rb"))

st.title("Loan Default Prediction")
st.write("Enter the loan details below to predict the default status.")

# Input fields
LNAMOUNT = st.slider("Loan Amount", min_value=1000.0, max_value=1000000.0, step=1000.0)
LNINTRATE = st.slider("Interest Rate", min_value=0.0, max_value=30.0, step=0.1)
LNINSTAMT = st.slider("Installment Amount", min_value=100.0, max_value=50000.0, step=100.0)
LNPAYFREQ = st.selectbox("Payment Frequency", ['2', '5', '12'])
QSPURPOSEDES = st.selectbox("Purpose of Loan", ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
LNBASELDESC = st.selectbox("Loan Type", ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
SEX = st.selectbox("Gender", ["M", "F"])
AGE = st.slider("Age", min_value=18, max_value=80, step=1)
CREDIT_CARD_USED = st.selectbox("Credit Card Used", ["Yes", "No"])
DEBIT_CARD_USED = st.selectbox("Debit Card Used", ["Yes", "No"])
QS_SECTOR = st.selectbox("Sector", ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 
    'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 
    'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])
LNPERIOD_CATEGORY = st.selectbox("Loan Period Category", ['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM'])
AVERAGE_SAGBAL = st.slider("Average Savings Balance", min_value=0.0, max_value=1000000.0, step=500.0)

# Create input DataFrame
input_data = pd.DataFrame({
    "LNAMOUNT": [LNAMOUNT],
    "LNINTRATE": [LNINTRATE],
    "LNINSTAMT": [LNINSTAMT],
    "LNPAYFREQ": [LNPAYFREQ],
    "QSPURPOSEDES": [QSPURPOSEDES],
    "LNBASELDESC": [LNBASELDESC],
    "SEX": [SEX],
    "AGE": [AGE],
    "CREDIT_CARD_USED": [CREDIT_CARD_USED],
    "DEBIT_CARD_USED": [DEBIT_CARD_USED],
    "QS_SECTOR": [QS_SECTOR],
    "LNPERIOD_CATEGORY": [LNPERIOD_CATEGORY],
    "AVERAGE_SAGBAL": [AVERAGE_SAGBAL]
})

# One-hot encode categorical variables
input_data = pd.get_dummies(input_data, columns=["QSPURPOSEDES", "QS_SECTOR", "LNBASELDESC","SEX","LNPAYFREQ", 'CREDIT_CARD_USED','DEBIT_CARD_USED',"LNPERIOD_CATEGORY"], drop_first=True)

# Ensure all required columns exist
missing_cols = set(X_train_ref.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0  # Add missing columns with default value 0

# Ensure column order matches training data
input_data = input_data[X_train_ref.columns]

# Standardize numerical features
num_features = ["LNAMOUNT", "LNINSTAMT", "AVERAGE_SAGBAL", "AGE", "LNINTRATE"]
input_data[num_features] = scaler.transform(input_data[num_features])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Default" if prediction[0] == 1 else "No Default"
    st.write(f"### Prediction: {result}")

# Reset button
def reset_inputs():
    st.experimental_rerun()

st.button("Reset", on_click=reset_inputs)
