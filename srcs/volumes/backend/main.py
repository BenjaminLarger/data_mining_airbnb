import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from parse import open_and_parse_file
from random_forest import random_forest_model
from variable_selector import select_pertinent_data
from gradient_boosting import gradient_boosting_model
from dictionary import get_and_parse_city
from knn import knn_model
from utils import compare_models
from tune_ridge import tune_ridge_model
app = Flask(__name__)

CORS(app)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return "Airbnb data analysis Backend is running!"

@app.route('/api/postdata', methods=['POST'])
def post_data():
    # Get user input from the request
    data = request.get_json()
    city = data.get('city', '')
    dictionary_response = get_and_parse_city(city)
    latitude = dictionary_response.get('latitude')
    longitude = dictionary_response.get('longitude')
    
    if not city:
        return jsonify({"error": "No city provided"}), 400
    
		# Open and parse the file
    data = open_and_parse_file(city)

    if data.get('error'):
        return jsonify(data), 400
    
		# Select pertinent data
    selected_data = select_pertinent_data(data)
    
		# Train models
    # random_forest_results = random_forest_model(selected_data, data)
    gradient_boosting_results = gradient_boosting_model(selected_data, data)
    linear_gradient_results = tune_ridge_model(selected_data, data)
    knn_results = knn_model(selected_data, data)
    
		# Simulate results
    random_forest_results = {'MAE_TRAIN': 337.26, 'MAE_TEST': 418.09, 'RMSE_TRAIN': 641.01, 'RMSE_TEST': 725.75, 'R2_TRAIN': 0.61, 'R2_TEST': 0.42, 'Cross-Validation RMSE': 815.06}
    # gradient_boosting_results = {'MAE_TRAIN': 423.06, 'MAE_TEST': 426.85, 'RMSE_TRAIN': 776.75, 'RMSE_TEST': 763.79, 'R2_TRAIN': 0.43, 'R2_TEST': 0.4, 'Cross-Validation RMSE': 819.79}
    # linear_gradient_results = {'MAE_TRAIN': 2.50, 'MAE_TEST': 2.50, 'RMSE_TRAIN': 4.43, 'RMSE_TEST': 4.43, 'R2_TRAIN': 1, 'R2_TEST': 1, 'Cross-Validation RMSE': 5.65}
    # knn_results = {'MAE_TRAIN': 0, 'MAE_TEST': 70.04, 'RMSE_TRAIN': 0, 'RMSE_TEST': 137, 'R2_TRAIN': 1, 'R2_TEST': 1, 'Cross-Validation RMSE': 145.22}
    
		# Compare and plot the results
    compare_models(random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results)

    # Create a json response
    return jsonify({
				'city': city,
				'latitude': latitude,
				'longitude': longitude,
				'data': selected_data,
				'random_forest_results': random_forest_results,
				'gradient_boosting_results': gradient_boosting_results,
				'linear_gradient_results': linear_gradient_results,
				'knn_results': knn_results
		})
    
if __name__ == '__main__':
    app.run(debug=True)
