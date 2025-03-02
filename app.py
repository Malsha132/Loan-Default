import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
model_path = 'classification_model_personal.pkl'
scaler_path = 'c_scaler.pkl'
columns_path = 'c_X_train.pkl'  # To ensure feature consistency

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(columns_path, 'rb') as file:
    train_columns = pickle.load(file).columns.tolist()

# Define categorical options
qspurpose_options = ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT']
lnbaseldesc_options = ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED']
sex_options = ['M', 'F']
credit_card_options = ['Yes', 'No']
debit_card_options = ['Yes', 'No']
qs_sector_options = ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV']

# Preprocessing function
def preprocess_input(user_input):
    input_data = pd.DataFrame([user_input])
    
    # Define categorical features
    categorical_features = ['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED']
    
    # One-hot encoding
    input_data = pd.get_dummies(input_data, columns=categorical_features)
    
    # Align with training data columns (fill missing columns with 0)
    input_data = input_data.reindex(columns=train_columns, fill_value=0)
    
    # Scale numerical features
    num_features = ['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']
    input_data[num_features] = scaler.transform(input_data[num_features])
    
    return input_data

# Streamlit App
st.title("Loan Default Risk Prediction")
st.write("Enter loan details to predict the risk of default.")

# User input fields
user_input = {
    'QSPURPOSEDES': st.selectbox("Loan Purpose", qspurpose_options),
    'LNAMOUNT': st.slider("Loan Amount", 1000, 1000000, 50000),
    'LNINSTAMT': st.slider("Installment Amount", 500, 50000, 5000),
    'LNBASELDESC': st.selectbox("Customer Segment", lnbaseldesc_options),
    'SEX': st.selectbox("Gender", sex_options),
    'AGE': st.slider("Age", 18, 80, 30),
    'CREDIT_CARD_USED': st.selectbox("Credit Card Used", credit_card_options),
    'DEBIT_CARD_USED': st.selectbox("Debit Card Used", debit_card_options),
    'QS_SECTOR': st.selectbox("Sector", qs_sector_options),
    'AVERAGE_SAGBAL': st.slider("Avg Savings Balance", 0, 1000000, 10000),
    'LNINTRATE': st.slider("Interest Rate", 0.1, 20.0, 5.0)
}

# Prediction button
if st.button("Predict Loan Default Risk"):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)[0]
    result = "Default" if prediction == 1 else "Not Default"
    st.write(f"### Prediction: {result}")

# Reset button
if st.button("Reset Inputs"):
    st.experimental_rerun()
