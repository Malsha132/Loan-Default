import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cache loading of models to optimize performance
@st.cache_resource
def load_model():
    return joblib.load('classification_model_personal.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('c_scaler.pkl')

@st.cache_resource
def load_columns():
    return joblib.load('c_X_train.pkl')

# Load the trained model, scaler, and reference feature names
model = load_model()
scaler = load_scaler()
columns = load_columns()

# Set Streamlit page title
st.title("Loan Default Risk Prediction")

# Dropdowns for categorical variables
sex = st.selectbox('Sex', ['M', 'F'])
qs_sector = st.selectbox('Sector', ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTICS',
                                    'FINANCIAL', 'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION',
                                    'TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE',
                                    'TRADERS', 'AGRICULTURE & FISHING',
                                    'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])
qspurposedes = st.selectbox('Loan Purpose', ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS',
                                             'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
lnbaseldesc = st.selectbox('Loan Base Description', ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE',
                                                     'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
lnpayfreq = st.selectbox('Payment Frequency', [2, 5, 12])
credit_card_used = st.selectbox('Credit Card Used', ['No', 'Yes'])
debit_card_used = st.selectbox('Debit Card Used', ['No', 'Yes'])
lnperiod_category = st.selectbox('Loan Period Category', ['SHORT-TERM', 'MEDIUM-TERM'])

# Slider for numerical inputs
age = st.slider('Age', min_value=18, max_value=80, value=30)
lnamount = st.number_input('Loan Amount', min_value=1000, max_value=1000000, value=50000)
lninstamt = st.number_input('Installment Amount', min_value=100, max_value=10000, value=1000)
lnintrate = st.number_input('Interest Rate', min_value=0.1, max_value=50.0, value=12.0)
averagesagbal = st.number_input('Average Savings Balance', min_value=0, max_value=1000000, value=20000)

# Prepare user input into a DataFrame
input_data = pd.DataFrame([[lnamount, lnintrate, lninstamt, age, averagesagbal, qspurposedes, qs_sector,
                            lnbaseldesc, sex, lnpayfreq, credit_card_used, debit_card_used, lnperiod_category]],
                          columns=['LNAMOUNT', 'LNINTRATE', 'LNINSTAMT', 'AGE', 'AVERAGE_SAGBAL', 'QSPURPOSEDES',
                                   'QS_SECTOR', 'LNBASELDESC', 'SEX', 'LNPAYFREQ', 'CREDIT_CARD_USED',
                                   'DEBIT_CARD_USED', 'LNPERIOD_CATEGORY'])

# Convert categorical values to match training data
#input_data['CREDIT_CARD_USED'] = input_data['CREDIT_CARD_USED'].map({'No': 0, 'Yes': 1})
#input_data['DEBIT_CARD_USED'] = input_data['DEBIT_CARD_USED'].map({'No': 0, 'Yes': 1})

# One-hot encode categorical variables
input_data_encoded = pd.get_dummies(input_data, columns=['QSPURPOSEDES', 'QS_SECTOR', 'LNBASELDESC', 'SEX',
                                                         'LNPAYFREQ', 'CREDIT_CARD_USED',
                                   'DEBIT_CARD_USED','LNPERIOD_CATEGORY'], drop_first=True)

# Ensure input matches training feature set
input_data_encoded = input_data_encoded.reindex(columns=columns, fill_value=0)

# Scale the numerical features
num_features = ['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']
input_data_encoded[num_features] = scaler.transform(input_data_encoded[num_features])

# Make prediction
prediction = model.predict(input_data_encoded)[0]

# Display prediction result
if prediction == 1:
    st.write("The loan default risk is HIGH.")
else:
    st.write("The loan default risk is LOW.")

# Reset button to refresh the page
if st.button('Reset'):
    st.experimental_rerun()
