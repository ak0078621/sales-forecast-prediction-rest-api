import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Store Sales Forecast Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for property features
productWeight = st.number_input("Product Weight", min_value=0.0, max_value=100.0, step=1.0, value=90.0)
productSugarContent = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
productAllocatedArea = st.number_input("Product Allocated Area", min_value=0.0, step=1, value=2)
productType = st.selectbox("Product Type", ["Frozen Foods", "Dairy", "Canned", "Baking Goods",
       "Health and Hygiene", "Snack Foods", "Meat", "Household",
       "Hard Drinks", "Fruits and Vegetables", "Breads", "Soft Drinks",
       "Breakfast", "Others", "Starchy Foods", "Seafood"])
productMRP = st.number_input("Product MRP", min_value=0.0, step=1, value=2) 
storeId = st.selectbox("Stores", ["OUT004", "OUT003", "OUT001", "OUT002"])
storeEstablishmentYear = st.selectbox("Store Year of Establishment", options=range(1950, 2027), index=24)
storeSize = st.selectbox("Store size", ["High", "Medium", "Low"])
storeLocationCityType = st.selectbox("Store Location City Type", ["Tier 1","Tier 2","Tier 3"])
storeType = st.selectbox("Store Type", ["Departmental Store","Supermarket Type1","Supermarket Type2","Food Mart"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': productWeight,
    'Product_Sugar_Content': productSugarContent,
    'Product_Allocated_Area': productAllocatedArea,
    'Product_Type': productType,
    'Product_MRP': productMRP,
    'Store_Id': storeId,                          
    'Store_Establishment_Year': storeEstablishmentYear,
    'Store_Size': storeSize,
    'Store_Location_City_Type': storeLocationCityType,
    'Store_Type': storeType
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales-forecast", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales Forecasr']
        st.success(f"Predicted Sales Forecast: {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/sales-forecast-batch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
