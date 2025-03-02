import streamlit as st
import joblib
import pandas as pd
#import numpy as np

# Load the trained model, scaler, and features
model = joblib.load('classification_model_personal.pkl')
scaler = joblib.load('c_scaler.pkl')
#columns = joblib.load('c_X_train.pkl')

# Set Streamlit page title
st.title("Loan Default Risk Prediction")
# Dropdowns for categorical variables
sex = st.selectbox('Sex', ['M', 'F'])
qs_sector = st.selectbox('Sector', ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 
                                   'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 
                                   'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 
                                   'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])
qspurposedes = st.selectbox('Loan Purpose', ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 
                                             'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
lnbaseldesc = st.selectbox('Loan Base Description', ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 
                                                   'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
lnpayfreq = st.selectbox('Payment Frequency', ['2', '5', '12'])
credit_card_used = st.selectbox('Credit Card Used', ['Yes', 'No'])
debit_card_used = st.selectbox('Debit Card Used', ['Yes', 'No'])
lnperiod_category = st.selectbox('Loan Period Category', ['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM'])

# Slider for numerical inputs
age = st.slider('Age', min_value=18, max_value=80, value=30)
lnamount = st.number_input('Loan Amount', min_value=1000, max_value=1000000, value=50000)
lninstamt = st.number_input('Installment Amount', min_value=100, max_value=10000, value=1000)
lnintrate = st.number_input('Interest Rate', min_value=0.1, max_value=50.0, value=12.0)
averagesagbal = st.number_input('Average Savings Balance', min_value=0, max_value=1000000, value=20000)

# Prepare user input into a DataFrame
input_data = pd.DataFrame([[lnamount,lnintrate,lninstamt, age,averagesagbal,qspurposedes, qs_sector, lnbaseldesc,sex, lnpayfreq,credit_card_used, debit_card_used, 
                             lnperiod_category]],
                          columns=['LNAMOUNT','LNINTRATE','LNINSTAMT', 'AGE','AVERAGE_SAGBAL','QSPURPOSEDES',  'QS_SECTOR','LNBASELDESC', 'SEX','LNPAYFREQ',  'CREDIT_CARD_USED', 'DEBIT_CARD_USED',
                                    'LNPERIOD_CATEGORY'])

# One-hot encode the categorical variables (like in your training data)
input_data_encoded = pd.get_dummies(input_data, columns=['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX', 
                                                         'LNPAYFREQ', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED', 
                                                         'LNPERIOD_CATEGORY'], drop_first=True)

# Ensure the columns match the ones used in training
input_data_encoded = input_data_encoded.reindex(columns=columns, fill_value=0)

# Scale the numerical features
input_data_scaled = scaler.transform(input_data_encoded[['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']])
input_data_encoded[['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']] = input_data_scaled
# Make prediction
prediction = model.predict(input_data_encoded)

# Display prediction result
if prediction == 1:
    st.write("The loan default risk is HIGH.")
else:
    st.write("The loan default risk is LOW.")
if st.button('Reset'):
    st.experimental_rerun()







