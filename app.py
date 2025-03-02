import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and columns list
model = joblib.load("classification_model_personal.pkl")
scaler = joblib.load("c_scaler.pkl")
columns = joblib.load("c_X_train.pkl")

# Streamlit app title
st.title('Loan Default Risk Prediction')

# Input form for loan details
loan_amount = st.slider('Loan Amount', 1000, 100000, 10000)
interest_rate = st.slider('Interest Rate (%)', 1.0, 25.0, 5.0)
loan_period = st.slider('Loan Period (Months)', 1, 60, 12)
installment_amount = st.slider('Installment Amount', 100, 5000, 300)
payment_frequency = st.selectbox('Payment Frequency', ['Monthly', 'Quarterly', 'Yearly'])
sector = st.selectbox('Sector', ['CONSUMPTION', 'TOURISM', 'CONSTRUCTION', 'HEALTHCARE', 'MANUFACTURING', 'TECHNOLOGY'])
age = st.slider('Age', 18, 80, 30)
average_savings_balance = st.slider('Average Savings Balance', 0, 100000, 5000)
credit_card_used = st.selectbox('Credit Card Used', [True, False])
debit_card_used = st.selectbox('Debit Card Used', [True, False])

# Create DataFrame from user input
input_data = pd.DataFrame({
    'Loan Amount': [loan_amount],
    'Interest Rate': [interest_rate],
    'Loan Period': [loan_period],
    'Installment Amount': [installment_amount],
    'Payment Frequency': [payment_frequency],
    'Sector': [sector],
    'Age': [age],
    'Average Savings Balance': [average_savings_balance],
    'Credit Card Used': [credit_card_used],
    'Debit Card Used': [debit_card_used]
})

# Apply one-hot encoding to categorical columns
input_data_encoded = pd.get_dummies(input_data, columns=['Payment Frequency', 'Sector'], drop_first=True)

# Ensure the same columns as the training data
missing_cols = set(columns) - set(input_data_encoded.columns)
for col in missing_cols:
    input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[columns]

# Apply scaling to the numerical columns
input_data_scaled = scaler.transform(input_data_encoded[['Loan Amount', 'Interest Rate', 'Installment Amount', 'Age', 'Average Savings Balance']])

# Add scaled numerical values back to the DataFrame
input_data_encoded[['Loan Amount', 'Interest Rate', 'Installment Amount', 'Age', 'Average Savings Balance']] = input_data_scaled

# Make a prediction using the trained model
prediction = model.predict(input_data_encoded)

# Display prediction result
if prediction == 1:
    st.write('Prediction: Loan Default Risk (Yes)')
else:
    st.write('Prediction: No Loan Default Risk')

# Reset button functionality
if st.button('Reset'):
    st.experimental_rerun()
