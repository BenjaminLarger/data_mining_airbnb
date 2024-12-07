import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from parse import open_and_parse_file
from random_forest import random_forest_model
from correlation import select_pertinent_data
from gradient_boosting import gradient_boosting_model
from regression import tune_ridge_model
from dictionary import get_and_parse_city
from knn import knn_model
from utils import build_json_response, compare_models

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
    logging.info(f"city: {city}")
    if not city:
        return jsonify({"error": "No city provided"}), 400
    
		# Open and parse the file
    data = open_and_parse_file(city)

    if data.get('error'):
        return jsonify(data), 400
    
		# Select pertinent data
    selected_data = select_pertinent_data(data)
    
		# Train models
    random_forest_results = random_forest_model(selected_data, data)
    # gradient_boosting_results = gradient_boosting_model(selected_data, data)
    # linear_gradient_results = tune_ridge_model(selected_data, data)
    # knn_results = knn_model(selected_data, data)
    
		# Simulate results
    # random_forest_results = {'MAE (Train)': 0.2, 'MAE (Test)': 0.3, 'RMSE (Train)': 0.2, 'RMSE (Test)': 0.3, 'R² (Train)': 0.2, 'R² (Test)': 0.3, 'Cross-Validation RMSE': 0.3}
    gradient_boosting_results = {'MAE (Train)': 0.2, 'MAE (Test)': 0.3, 'RMSE (Train)': 0.2, 'RMSE (Test)': 0.3, 'R² (Train)': 0.2, 'R² (Test)': 0.3, 'Cross-Validation RMSE': 0.3}
    linear_gradient_results = {'MAE (Train)': 0.2, 'MAE (Test)': 0.3, 'RMSE (Train)': 0.2, 'RMSE (Test)': 0.3, 'R² (Train)': 0.2, 'R² (Test)': 0.3, 'Cross-Validation RMSE': 0.3}
    knn_results = {'MAE (Train)': 0.2, 'MAE (Test)': 0.3, 'RMSE (Train)': 0.2, 'RMSE (Test)': 0.3, 'R² (Train)': 0.2, 'R² (Test)': 0.3, 'Cross-Validation RMSE': 0.3}
    
		# Compare and plot the results
    compare_models(random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results)

    # Create a json response
    #json_response = build_json_response(city, dictionary_response, selected_data, random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results)
    #return json_response
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

    # # Get city, latitude, and longitude information from the dictionary
    # cities_info_json = {}
    # for city, country in cities_json_gemini.items():
    #     cities_info_json[city] = get_cities_from_dictionary(city, country)
    # logging.info(f"Cities Info JSON: {cities_info_json}")

    # return cities_info_json
    
if __name__ == '__main__':
    app.run(debug=True)
