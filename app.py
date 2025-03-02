import pickle
import pandas as pd

# Load the list
with open("c_X_train.pkl", "rb") as file:
    X_train_list = pickle.load(file)

# Manually define feature names (Use the same order as when training the model)
feature_names = [
    "LNAMOUNT", "LNINTRATE", "LNPERIOD", "LNINSTAMT", "LNPAYFREQ", "QSPURPOSEDES_CONSTRUCTION", 
    "QSPURPOSEDES_EDUCATION", "QSPURPOSEDES_INVESTMENT", "QSPURPOSEDES_PERSONAL NEEDS", 
    "QSPURPOSEDES_PURCHASE OF PROPERTY", "QSPURPOSEDES_PURCHASE OF VEHICLE", 
    "QSPURPOSEDES_WORKING CAPITAL REQUIREMENT", "LNBASELDESC_FINANCIAL INSTITUTIONS",
    "LNBASELDESC_INDIVIDUALS", "LNBASELDESC_MICRO FINANCE", "LNBASELDESC_MIDDLE MARKET CORPORATES",
    "LNBASELDESC_SME", "LNBASELDESC_UNCLASSIFIED", "SEX_M", "SEX_F", "AGE", 
    "CREDIT_CARD_USED_Yes", "CREDIT_CARD_USED_No", "DEBIT_CARD_USED_Yes", "DEBIT_CARD_USED_No",
    "QS_SECTOR_OTHER SERVICES", "QS_SECTOR_CONSUMPTION", "QS_SECTOR_MANUFACTURING & LOGISTIC",
    "QS_SECTOR_FINANCIAL", "QS_SECTOR_CONSTRUCTION & INFRASTRUCTURE", "QS_SECTOR_EDUCATION",
    "QS_SECTOR_TECHNOLOGY & INNOVATION", "QS_SECTOR_TOURISM", "QS_SECTOR_HEALTHCARE", 
    "QS_SECTOR_TRADERS", "QS_SECTOR_AGRICULTURE & FISHING", "QS_SECTOR_PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV"
]

# Convert to DataFrame
if isinstance(X_train_list, list):
    X_train_df = pd.DataFrame(X_train_list, columns=feature_names)
else:
    raise ValueError("X_train is not in the correct format.")

# Now you can use X_train_df in your model
