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

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return "Airbnb data analysis Backend is running!"

def build_json_response(city, dictionary_response, selected_data, random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results):
		random_forest_r2 = random_forest_results.get('R^2 Score')
		random_forest_rmse = random_forest_results.get('Root Mean Squared Error')
		gradient_boosting_r2 = gradient_boosting_results.get('R^2 Score')
		gradient_boosting_rmse = gradient_boosting_results.get('Root Mean Squared Error')
		linear_regression_r2 = linear_gradient_results.get('R-squared')
		linear_regression_rmse = linear_gradient_results.get('Root Mean Squared Error')
		knn_r2 = knn_results.get('R^2 Score')
		knn_rmse = knn_results.get('Root Mean Squared Error')
		return {
				'city': city,
				'latitude': dictionary_response.get('latitude'),
				'longitude': dictionary_response.get('longitude'),
				'data': selected_data,
				'random_forest_r2': random_forest_r2,
				'random_forest_rmse': random_forest_rmse,
				'gradient_boosting_r2': gradient_boosting_r2,
				'gradient_boosting_rmse': gradient_boosting_rmse,
				'linear_regression_r2': linear_regression_r2,
				'linear_regression_rmse': linear_regression_rmse,
				'knn_r2': knn_r2,
				'knn_rmse': knn_rmse
		}

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
    
    data = open_and_parse_file(city)

    if data.get('error'):
        return jsonify(data), 400
    
    selected_data = select_pertinent_data(data)
    
    #random_forest_results = random_forest_model(selected_data, data)
    # gradient_boosting_results = gradient_boosting_model(selected_data, data)
    # linear_gradient_results = tune_ridge_model(selected_data, data)
    knn_results = knn_model(selected_data, data)
    logging.info(f"KNN Results: {knn_results}")
    random_forest_results = {'R^2 Score': 0.29, 'Root Mean Squared Error': 0.3}
    gradient_boosting_results = {'R^2 Score': 0.2, 'Root Mean Squared Error': 0.3}
    # knn_results = {'R^2 Score': 0.2, 'Root Mean Squared Error': 0.3}
    linear_gradient_results = {'R-squared': 0.3, 'Root Mean Squared Error': 0.3}
    # Create a json response
    json_response = build_json_response(city, dictionary_response, selected_data, random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results)
    return json_response

    # # Get city, latitude, and longitude information from the dictionary
    # cities_info_json = {}
    # for city, country in cities_json_gemini.items():
    #     cities_info_json[city] = get_cities_from_dictionary(city, country)
    # logging.info(f"Cities Info JSON: {cities_info_json}")

    # return cities_info_json
    
if __name__ == '__main__':
    app.run(debug=True)
