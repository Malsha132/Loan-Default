import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and feature columns
model = joblib.load('classification_model_personal.pkl')
scaler = joblib.load('c_scaler.pkl')
X_train_columns = joblib.load('c_X_train.pkl')

# Streamlit app UI
st.title('Loan Default Prediction')

# Collect user inputs for loan details
loan_amount = st.slider('Loan Amount', 1000, 100000, 10000)
interest_rate = st.slider('Interest Rate (%)', 1.0, 25.0, 5.0)
installment_amount = st.slider('Installment Amount', 100, 5000, 300)
payment_frequency = st.selectbox('Payment Frequency', ['Monthly', 'Quarterly', 'Yearly'])
sector = st.selectbox('Sector', ['CONSUMPTION', 'TOURISM', 'CONSTRUCTION', 'HEALTHCARE', 'MANUFACTURING', 'TECHNOLOGY'])
purpose = st.selectbox('Loan Purpose', ['Personal', 'Mortgage', 'Auto Loan'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
credit_card_used = st.selectbox('Credit Card Used', ['Yes', 'No'])
debit_card_used = st.selectbox('Debit Card Used', ['Yes', 'No'])
loan_period_category = st.selectbox('Loan Period Category', ['Short-term', 'Medium-term', 'Long-term'])

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'LNAMOUNT': [loan_amount],
    'LNINTRATE': [interest_rate],
    'LNINSTAMT': [installment_amount],
    'LNPAYFREQ': [payment_frequency],
    'QSPURPOSEDES': [purpose],
    'LNBASELDESC': [sector],
    'SEX': [gender],
    'AGE': [age],
    'CREDIT_CARD_USED': [credit_card_used],
    'DEBIT_CARD_USED': [debit_card_used],
    'QS_SECTOR': [sector],
    'LNPERIOD_CATEGORY': [loan_period_category],
    'AVERAGE_SAGBAL': [0]  # You can set this dynamically if needed
})

# One-hot encode categorical variables
input_data_encoded = pd.get_dummies(input_data, columns=['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX', 'LNPAYFREQ', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED', 'LNPERIOD_CATEGORY'], drop_first=True)

# Ensure that the columns of input match the trained columns
input_data_encoded = input_data_encoded.reindex(columns=X_train_columns, fill_value=0)

# Scale the numeric features
input_data_scaled = scaler.transform(input_data_encoded[['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']])

# Prepare the full input for prediction
input_data_encoded[['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']] = input_data_scaled

# Make prediction using the model
prediction = model.predict(input_data_encoded)

# Display the result
if prediction == 1:
    st.write('Prediction: Loan Default Risk (Yes)')
else:
    st.write('Prediction: No Loan Default Risk')

# Reset button to clear inputs
if st.button('Reset'):
    st.experimental_rerun()  # This will reload the app and reset the inputs
