import pickle
import pandas as pd
import streamlit as st

# Load the trained model and scaler
with open("classification_model_personal.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("c_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load expected feature names correctly
with open("c_X_train.pkl", "rb") as x_train_file:
    X_train = pickle.load(x_train_file)

# Ensure X_train is a DataFrame
if isinstance(X_train, pd.DataFrame):
    expected_columns = X_train.columns
else:
    raise ValueError("The loaded X_train is not a DataFrame. Please check the saved pickle file.")

# Streamlit App
st.title("Loan Default Prediction")

# User Input
LNAMOUNT = st.number_input("Loan Amount", min_value=0.0, step=5000.0)
LNINTRATE = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
LNPERIOD = st.number_input("Loan Period (Months)", min_value=1, step=1)
LNINSTAMT = st.number_input("Installment Amount", min_value=0.0, step=100.0)
LNPAYFREQ = st.selectbox("Payment Frequency", [1, 2])
QSPURPOSEDES = st.selectbox("Loan Purpose", ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
LNBASELDESC = st.selectbox("Customer Segment", ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
SEX = st.selectbox("Gender", ['M', 'F'])
AGE = st.slider("Age", 18, 80, 30)
CREDIT_CARD_USED = st.selectbox("Used Credit Card?", ['Yes', 'No'])
DEBIT_CARD_USED = st.selectbox("Used Debit Card?", ['Yes', 'No'])
QS_SECTOR = st.selectbox("Sector", ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])

# Convert categorical inputs into a DataFrame
input_data = pd.DataFrame([[LNAMOUNT, LNINTRATE, LNPERIOD, LNINSTAMT, LNPAYFREQ, QSPURPOSEDES, LNBASELDESC, SEX, AGE, CREDIT_CARD_USED, DEBIT_CARD_USED, QS_SECTOR]],
                          columns=['LNAMOUNT', 'LNINTRATE', 'LNPERIOD', 'LNINSTAMT', 'LNPAYFREQ', 'QSPURPOSEDES', 'LNBASELDESC', 'SEX', 'AGE', 'CREDIT_CARD_USED', 'DEBIT_CARD_USED', 'QS_SECTOR'])

# One-hot encode categorical variables
input_data_encoded = pd.get_dummies(input_data, drop_first=True)

# Align with expected model columns
input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

# Normalize numerical features
num_features = ['LNAMOUNT', 'LNINTRATE', 'LNPERIOD', 'LNINSTAMT', 'AGE']
input_data_encoded[num_features] = scaler.transform(input_data_encoded[num_features])

# Predict default status
if st.button("Predict Loan Default"):
    prediction = model.predict(input_data_encoded)[0]
    result = "Default" if prediction == 1 else "No Default"
    st.success(f"Prediction: {result}")

# Debugging (if needed)
st.write("Expected Columns:", expected_columns.tolist())
st.write("Input Data Columns:", input_data_encoded.columns.tolist())
