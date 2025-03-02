import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pickle files using joblib
model = joblib.load("classification_model_personal.pkl")
scaler = joblib.load("c_scaler.pkl")
X_train_ref = joblib.load("c_X_train.pkl")

# Extract column names for one-hot encoding
categorical_columns = ['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX', 'LNPAYFREQ', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED', 'LNPERIOD_CATEGORY']
numerical_columns = ['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']

# Streamlit UI
st.title("Loan Default Prediction App")
st.write("Enter loan details below to predict default risk.")

# Create input fields
lnamount = st.slider("Loan Amount", min_value=1000, max_value=1000000, step=1000)
lninstamt = st.slider("Installment Amount", min_value=100, max_value=50000, step=100)
avg_sagbal = st.slider("Average Savings Balance", min_value=0, max_value=500000, step=1000)
age = st.slider("Age", min_value=18, max_value=80, step=1)
lnintrate = st.slider("Interest Rate (%)", min_value=1.0, max_value=20.0, step=0.1)

# Dropdowns for categorical variables
qpurpose = st.selectbox("Loan Purpose", X_train_ref['QSPURPOSEDES'].unique())
qsector = st.selectbox("Sector", X_train_ref['QS_SECTOR'].unique())
lnbase = st.selectbox("Loan Base", X_train_ref['LNBASELDESC'].unique())
sex = st.selectbox("Sex", X_train_ref['SEX'].unique())
lnpayfreq = st.selectbox("Payment Frequency", X_train_ref['LNPAYFREQ'].unique())
credit_card = st.selectbox("Credit Card Used", ["Yes", "No"])
debit_card = st.selectbox("Debit Card Used", ["Yes", "No"])
lnperiod_cat = st.selectbox("Loan Period Category", X_train_ref['LNPERIOD_CATEGORY'].unique())

# Process input
input_data = pd.DataFrame([[lnamount, lninstamt, avg_sagbal, age, lnintrate, qpurpose, qsector, lnbase, sex, lnpayfreq, credit_card, debit_card, lnperiod_cat]],
                          columns=numerical_columns + categorical_columns)
input_data[['CREDIT_CARD_USED', 'DEBIT_CARD_USED']] = input_data[['CREDIT_CARD_USED', 'DEBIT_CARD_USED']].replace({"Yes": 1, "No": 0})

# One-hot encoding
input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Align columns with training data
missing_cols = set(X_train_ref.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[X_train_ref.columns]

# Scale numerical features
input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Default" if prediction[0] == 1 else "Not Default"
    st.subheader(f"Prediction: {result}")

# Reset button
if st.button("Reset"):
    st.experimental_rerun()

