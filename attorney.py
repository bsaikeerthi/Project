import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Load the pre-trained logistic regression model
model = pickle.load(open('D:\DS project\DS pro 2\logistic_regression_model.pkl', 'rb'))

# Function to encode categorical variables
def encode_input(data):
    # Label Encoding for Policy_Type
    le = LabelEncoder()
    data['Policy_Type'] = le.fit_transform(data['Policy_Type'])

    # Ordinal Encoding for Accident_Severity and Driving_Record
    oeAS = OrdinalEncoder(categories=[['Minor', 'Moderate', 'Severe']])
    data['Accident_Severity'] = oeAS.fit_transform(data[['Accident_Severity']])

    oeDR = OrdinalEncoder(categories=[['Clean', 'Minor Offenses', 'Major Offenses']])
    data['Driving_Record'] = oeDR.fit_transform(data[['Driving_Record']])

    return data

# Streamlit UI for user input
st.title('Attorney Involvement in Claims Prediction')
st.write("Please enter the following details to predict attorney involvement in a claim:")

# Collecting user inputs
clmsex = st.selectbox('Claimant Gender (1 = Male, 0 = Female)', [0, 1])
clmage = st.number_input('Claimant Age', min_value=18, max_value=100, value=30)
loss = st.number_input('Financial Loss', min_value=0.0, value=5000.0)
accident_severity = st.selectbox('Accident Severity', ['Minor', 'Moderate', 'Severe'])
claim_approval_status = st.selectbox('Claim Approval Status (1 = Approved, 0 = Denied)', [0, 1])
policy_type = st.selectbox('Policy Type', ['Comprehensive', 'Third-Party'])
driving_record = st.selectbox('Driving Record', ['Clean', 'Minor Offenses', 'Major Offenses'])
claim_amount_requested = st.number_input('Claim Amount Requested', min_value=0.0, value=10000.0)
settlement_amount = st.number_input('Settlement Amount', min_value=0.0, value=2000.0)

# Feature engineering: Calculate settlement_ratio
settlement_ratio = settlement_amount / claim_amount_requested if claim_amount_requested != 0 else 0

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'CLMSEX': [clmsex],
    'CLMAGE': [clmage],
    'LOSS': [loss],
    'Accident_Severity': [accident_severity],
    'Claim_Approval_Status': [claim_approval_status],
    'Policy_Type': [policy_type],
    'Driving_Record': [driving_record],
    'settlement_ratio': [settlement_ratio]
})

# Encode categorical variables
input_data = encode_input(input_data)

# Make prediction
prediction = model.predict(input_data)

# Display prediction result
if prediction == 1:
    st.write("An attorney is likely to be involved in the claim.")
else:
    st.write("An attorney is unlikely to be involved in the claim.")
