# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_forecast_predictor_api = Flask("SuperKart Sales Forecast Predictor")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@sales_forecast_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Forecast Prediction API!"

# Define an endpoint for single property prediction (POST request)
@sales_forecast_predictor_api.post('/v1/sales-forecast')
def predict_sales_forecast():
    """
    This function handles POST requests to the '/v1/sales-forecast' endpoint.
    It expects a JSON payload containing product, store and sales details and returns
    the predicted sales forecast as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Sugar_Content': property_data['Product_Sugar_Content'],
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_Type': property_data['Product_Type_Category'],
        'Product_MRP': property_data['Product_MRP'],
        'Store_Establishment_Year': 2026 - property_data['Store_Age_Years'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get Product_Store_Sales_Total)
    predicted_store_sales = model.predict(input_data)[0]

    print(f"Raw model output (log): {predicted_store_sales}")

    # Convert predicted_price to Python float
    predicted_sales_forecast = round(float(predicted_store_sales), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted sales forecast': predicted_sales_forecast})


# Define an endpoint for batch prediction (POST request)
@sales_forecast_predictor_api.post('/v1/sales-forecast-batch')
def predict_sales_forecast_batch():
    file = request.files['file']
    input_data = pd.read_csv(file)

    # Transform columns to match what the model was trained on
    input_data['Store_Establishment_Year'] = 2026 - input_data['Store_Age_Years']
    input_data = input_data.rename(columns={'Product_Type_Category': 'Product_Type'})

    # Drop columns the model doesn't use
    input_data = input_data.drop(columns=['Store_Age_Years', 'Product_Id_char'], errors='ignore')

    predicted_store_sales = model.predict(input_data).tolist()
    predicted_sales = [round(float(sale), 2) for sale in predicted_store_sales]

    product_ids = list(range(1, len(input_data) + 1))
    output_dict = dict(zip(product_ids, predicted_sales))

    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_forecast_predictor_api.run(debug=True)
