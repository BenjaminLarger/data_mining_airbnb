import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from parse import open_and_parse_file

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
    logging.info(f"city: {city}")

    if not city:
        return jsonify({"error": "No city provided"}), 400
    
    data = open_and_parse_file(city)
    
    return jsonify({"message": "Data received successfully"}), 200

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
