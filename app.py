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
loan_amount = st.number_input('Enter Loan Amount (LKR):', min_value=0, step=0.001, format="%.3f", key="loan_amount")
interest_rate = st.number_input('Enter Loan Interest Rate (LKR):', min_value=0, step=0.001, format="%.3f", key="interest_rate")
installment_amount = st.number_input('Enter Loan Installment (LKR):', min_value=0, step=0.001, format="%.3f", key="installment_amount")
payment_frequency = st.selectbox('Select the Payement Frequency:', ['2', '5', '12'], key="payment_frequency")
credit_card_used = st.radio('Credict Card Used', ['Yes', 'No'], key="credit_card_used")
debit_card_used = st.radio('Debit Card Used', ['Yes', 'No'], key="debit_card_used")
sector = st.selectbox('Select the Customer Sector:', ['OTHER SERVICES','CONSUMPTION','MANUFACTURING & LOGISTIC','FINANCIAL','CONSTRUCTION & INFRASTRUCTURE','EDUCATION','TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE','TRADERS' ,'AGRICULTURE & FISHINIG','PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV' ], key="sector")
purpose = st.selectbox('Select the Loan Purpose:', ['CONSTRUCTION','EDUCATION','INVESTMENT','PERSONAL NEEDS','PURCHASE OF PROPERTY','PURCHASE OF VEHICLE','WORKING CAPITAL REQUIREMENT', key="purpose")
gender = st.selectbox('Select the Gender:', ['F','M' ], key="gender")
age = st.number_input('Enter Age:', min_value=0, step=0.001, format="%.3f", key="age")
loan_period_category = st.selectbox('Select the Loan Period:', ['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM' ], key="loan_period_category")
avg_sagbal = st.number_input('Average Saving Balance:', min_value=0, step=0.001, format="%.3f", key="avg_sagbal")
basel = st.selectbox('Select the Customer Group:', ['FINANCIAL INSTITUTIONS','INDIVIDUALS','MICRO FINANCE','MIDDLE MARKET CORPORATES','SME','UNCLASSIFIED'], key="basel")




# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'LNAMOUNT': [loan_amount],
    'LNINTRATE': [interest_rate],
    'LNINSTAMT': [installment_amount],
    'LNPAYFREQ': [payment_frequency],
    'QSPURPOSEDES': [purpose],
    'LNBASELDESC': [basel],
    'SEX': [gender],
    'AGE': [age],
    'CREDIT_CARD_USED': [credit_card_used],
    'DEBIT_CARD_USED': [debit_card_used],
    'QS_SECTOR': [sector],
    'LNPERIOD_CATEGORY': [loan_period_category],
    'AVERAGE_SAGBAL': [avg_sagbal]  # You can set this dynamically if needed
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
