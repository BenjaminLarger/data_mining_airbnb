import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from parse import open_and_parse_file
from random_forest import random_forest_model
from correlation import select_pertinent_data
from gradient_boosting import gradient_boosting_model
from regression import linear_regression_model
from dictionary import get_and_parse_city

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
    
    data = open_and_parse_file(city)

    if data.get('error'):
        return jsonify(data), 400
    
    selected_data = select_pertinent_data(data)
    
    #random_forest_model(selected_data, data)
    #gradient_boosting_model(selected_data, data)
    #linear_regression_model(selected_data, data)
    
    # Create a json response
    json_response = {
        'city': city,
        'latitude': latitude,
        'longitude': longitude,
        'data': selected_data
    }
    return json_response
    # # Call the Interrogator API
    # cities_json_gemini = get_cities(file)
    # if cities_json_gemini.get('error') or cities_json_gemini == {}:
    #     return jsonify(cities_json_gemini), 400
    # #cities = ['Sevilla', 'Malaga', 'Granada', 'Cordoba', 'Cadiz', 'Huelva', 'Jaen', 'Almeria']

    # # Get city, latitude, and longitude information from the dictionary
    # cities_info_json = {}
    # for city, country in cities_json_gemini.items():
    #     cities_info_json[city] = get_cities_from_dictionary(city, country)
    # logging.info(f"Cities Info JSON: {cities_info_json}")

    # return cities_info_json
    
if __name__ == '__main__':
    app.run(debug=True)
