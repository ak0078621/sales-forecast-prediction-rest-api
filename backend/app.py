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
        'Store_Establishment_Year': property_data['Store_Age_Years'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get Product_Store_Sales_Total)
    predicted_store_sales = model.predict(input_data)[0]

    print(f"Raw model output (log): {predicted_store_sales}")

    # Calculate actual price
    predicted_sales = np.exp(predicted_store_sales)

    print(f"After exp: {predicted_sales}")

    # Convert predicted_price to Python float
    predicted_sales_forecast = round(float(predicted_sales), 2)
    # The conversion above is needed as we convert the model prediction (log price) to actual price using np.exp, which returns predictions as NumPy float32 values.
    # When we send this value directly within a JSON response, Flask's jsonify function encounters a datatype error

    # Return the actual price
    return jsonify({'Predicted sales forecast': predicted_sales_forecast})


# Define an endpoint for batch prediction (POST request)
@sales_forecast_predictor_api.post('/v1/sales-forecast-batch')
def predict_sales_forecast_batch():
    """
    This function handles POST requests to the '/v1/sales-forecast-batch' endpoint.
    It expects a CSV file containing product, store and sales details details for multiple stores
    and returns the predicted sales forecast for stores as a dictionary in the JSON response. 
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_store_sales = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_sales = [round(float(np.exp(product_store_sales)), 2) for product_store_sales in predicted_store_sales]

    # Create a dictionary of predictions with property IDs as keys
    product_ids = input_data['id'].tolist()  # Assuming 'id' is the property ID column
    output_dict = dict(zip(product_ids, predicted_sales))  # Use actual prices

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_forecast_predictor_api.run(debug=True)
