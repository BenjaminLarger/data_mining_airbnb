import csv, logging, os
logging.basicConfig(level=logging.INFO)
# Path to your CSV file
CSV_FILE_PATH = '/build/utils/worldcities.csv'

def highlight_differences(str1, str2):
    differences = []
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            differences.append(f"Position {i}: '{char1}' != '{char2}'")
    if len(str1) != len(str2):
        longer_str = str1 if len(str1) > len(str2) else str2
        for i in range(len(str1), len(str2)):
            differences.append(f"Position {i}: '{longer_str[i]}' (extra character)")
    return differences


def get_cities_from_dictionary(city, country):
  """
  Returns city information (name, latitude, longitude) for a given city name.

  Args:
    city (str): Name of the city
    country (str): Name of the country

  """
  logging.info(f"Searching for city: {city}, in country: {country}")
  city_name = city.strip().lower()
  country_name = country.strip().lower()

  with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
      if row['city_ascii'].strip().lower() == city_name:
        difference = highlight_differences(row['country'].strip().lower(), country_name)
        if difference:
          logging.info(f"Country name difference: {difference}, for city: {city_name}, in country: {row['country'].strip().lower()}")
      if row['city_ascii'].strip().lower() == city_name.strip() and row['country'].strip().lower() == country_name.strip():
        return {
          'latitude': float(row['lat']),
          'longitude': float(row['lng'])
        }

def get_and_parse_city(city):
	"""
	Returns city information (name, latitude, longitude) for a given city name.

	Args:
		city (str): Name of the city
	"""
	logging.info(f"Searching for city: {city}")
	if city == 'rio':
		city_dic = 'Rio de Janeiro'
		country = 'Brazil'
	elif city == 'paris':
		city_dic = city
		country = 'france'
	elif city == 'rome':
		city_dic = city
		country = 'italy'
	elif city == 'istanbul':
		city_dic = city
		country = 'turkey'
	elif city == 'sydney':
		city_dic = 'Sydney'
		country = 'Australia'
	elif city == 'new-york':
		city_dic = 'New York'
		country = 'United States'
	elif city == 'mexico':
		city_dic = 'mexico city'
		country = 'mexico'
	elif city == 'london':
		city_dic = city
		country = 'United Kingdom'
	return get_cities_from_dictionary(city_dic, country)
