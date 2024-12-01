import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dictionary import get_cities_from_dictionary
from gemini import get_cities

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return "Travel Recommender Backend is running!"

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    # Get user input from the request
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Call the Interrogator API
    cities_json_gemini = get_cities(query)
    if cities_json_gemini.get('error') or cities_json_gemini == {}:
        return jsonify(cities_json_gemini), 400
    #cities = ['Sevilla', 'Malaga', 'Granada', 'Cordoba', 'Cadiz', 'Huelva', 'Jaen', 'Almeria']

    # Get city, latitude, and longitude information from the dictionary
    cities_info_json = {}
    for city, country in cities_json_gemini.items():
        cities_info_json[city] = get_cities_from_dictionary(city, country)
    logging.info(f"Cities Info JSON: {cities_info_json}")

    return cities_info_json
    
if __name__ == '__main__':
    app.run(debug=True)
