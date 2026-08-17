# Import necessary libraries
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

sales_forecast_predictor_api = Flask("SuperKart Sales Forecast Predictor")

model = joblib.load("superkart_model.joblib")

@sales_forecast_predictor_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Forecast Prediction API!"

@sales_forecast_predictor_api.post('/v1/sales-forecast')
def predict_sales_forecast():
    property_data = request.get_json()

    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Sugar_Content': property_data['Product_Sugar_Content'],
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_Type': property_data['Product_Type_Category'],           # rename
        'Product_MRP': property_data['Product_MRP'],
        'Store_Establishment_Year': 2026 - property_data['Store_Age_Years'],  # convert age → year
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type']
    }

    input_data = pd.DataFrame([sample])

    predicted_store_sales = model.predict(input_data)[0]

    predicted_sales_forecast = round(float(predicted_store_sales), 2)  # no np.exp()

    return jsonify({'Predicted sales forecast': predicted_sales_forecast})


@sales_forecast_predictor_api.post('/v1/sales-forecast-batch')
def predict_sales_forecast_batch():
    file = request.files['file']
    input_data = pd.read_csv(file)

    # Save product IDs before dropping
    product_ids = input_data['Product_Id_char'].tolist()

    # Transform columns to match model training
    input_data['Store_Establishment_Year'] = 2026 - input_data['Store_Age_Years']
    input_data = input_data.rename(columns={'Product_Type_Category': 'Product_Type'})

    # Drop columns the model doesn't use
    input_data = input_data.drop(columns=['Store_Age_Years', 'Product_Id_char'], errors='ignore')

    predicted_store_sales = model.predict(input_data).tolist()
    predicted_sales = [round(float(sale), 2) for sale in predicted_store_sales]

    output_dict = dict(zip(product_ids, predicted_sales))
    return output_dict


if __name__ == '__main__':
    sales_forecast_predictor_api.run(debug=True)
