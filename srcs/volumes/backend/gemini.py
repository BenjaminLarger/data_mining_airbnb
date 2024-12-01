import logging, requests, os
import google.generativeai as genai
logging.basicConfig(level=logging.INFO)

# Get GEMENI_API_KEY from the environment

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def parse_gemini_prompt(prompt):
    """
    Parse the response from the GEMINI model.

    Response format:
    [
      {'Barcelona': 'Spain'},
      {'Lisbon': 'Portugal'},
      {'Nice': 'France'},
      {'Athens': 'Greece'},
      {'Valletta': 'Malta'}
    ]
    """
    try:
        # Loop until the end of the response
        cities = []
        countries = []
        for line in prompt.split("{"):
            line = line.strip().replace("{", "")
            line = line.strip().replace("}", "")
            logging.info(f"Line: {line}")
            # if the line has no coma, skip it
            if not "]" in line and not "," in line:
                continue
            # Split the line by "'"
            city, country = line.split(":")
            # Trim the city and country
            city = city.strip().replace("'", "")
            city = city.strip().replace("}", "")
            city = city.strip().replace(",", "")
            city = city.strip().replace("\"", "")
            country = country.strip().replace("}", "")
            country = country.strip().replace("'", "")
            country = country.strip().replace(",", "")
            country = country.strip().replace("\"", "")
            country = country.strip().replace("\n", "")
            country = country.strip().replace("]", "")
            country = country.strip().replace("`", "")
            logging.info(f"City: {city}, Country: {country}")
            # Append the city and country to the lists
            cities.append(city)
            countries.append(country)
        # Create a list of dictionaries
        cities_json = {}
        for i in range(len(cities)):
            cities_json[cities[i]] = countries[i]
        logging.info(f"Cities JSON: {cities_json}")
        return cities_json 
    except Exception as e:
        logging.error(f"Error parsing the response: {e}")
        return {"error": str(e)}
       

def get_cities(user_input):
    try:
        logging.info(f"API key: {os.environ['GEMINI_API_KEY']}")
        # Print API key for debugging

        # Prepare the API request
        prompt = (
            "Generate a list of cities in english and their countries based on the input. "
            "Return the result in this JSON format:\n"
            "[{'CityName': 'CountryName'},"
            "{'CityName': 'CountryName'},"
            " ...]"
            " Do not send any other information in the response.\n"
            f"Input: {user_input}"
        )

        logging.info(f"Prompt: {prompt}")
        response = model.generate_content(prompt)
        logging.info(f"Response gemeni: {response.text}")
        # response = "```json [{'Barcelona': 'Spain'}, {'Lisbon': 'Portugal'}, {'Athens': 'Greece'}, {'Valletta': 'Malta'}, {'Nice': 'France'}]"
        # logging.info(f"Response: {response}")

        # Example response
        # response = """
        # [
        #   {'Barcelona': 'Spain'},
        #   {'Lisbon': 'Portugal'},
        #   {'Sevilla': 'Spain'},
        #   {'Athens': 'Greece'},
        #   {'Valletta': 'Malta'}
        # ]
        # """

        # Parse the response
        if (response.text):
          parsed_response = parse_gemini_prompt(response.text)
          return parsed_response
        else:
          return {"error": "No response from the model"}
        # if (response):
        #   parsed_response = parse_gemini_prompt(response)
        #   return parsed_response
        # else:
        #   return {"error": "No response from the model"}

    except Exception as e:
        logging.error(f"Error calling Interrogator API: {e}")
        return {"error": str(e)}