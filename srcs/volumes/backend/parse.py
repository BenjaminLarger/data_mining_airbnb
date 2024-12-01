import logging, requests, os
import pandas as pd
from fastparquet import ParquetFile

logging.basicConfig(level=logging.INFO)

FILE_DIR = "build/utils/subject/"

def	clean_data(df):
	# clean_dataset init
		"""
		Cleans the given DataFrame by filtering rows based on specific criteria.

		Parameters:
		df (pandas.DataFrame): The input dataset to clean.

		Returns:
		pandas.DataFrame: The cleaned dataset.
		"""
		# import numpy as np

		# Define valid values for categorical columns
		logging.info('===============================')
		valid_property_types = [
				'Entire rental unit',
				'Private room in rental unit',
				'Room in hotel',
				'Entire home',
				'Private room in bed and breakfast',
				'Entire condo'
		]
		valid_room_types = ['Private room', 'Entire home/apt', 'Hotel room', 'Shared room']
		# valid_neighbourhood_cleansed = ['Le Marais', 'Popincourt', 'Buttes-Chaumont', 'Vaugirard', 'Opéra', 'Entrepôt', 'Batignolles-Monceau', 'Passy', 'Reuilly', 'Gobelins', 'Luxembourg', 'Palais-Bourbon', 'Ménilmontant', 'Observatoire', 'Panthéon', 'Hôtel-de-Ville', 'Bourse', 'Louvre']
		# Filter rows based on conditions
		df['host_is_superhost'] = df['host_is_superhost'].map({'f': 0, 't': 1})
		df['instant_bookable'] = df['instant_bookable'].map({'f': 0, 't': 1})
		
		clean_dataset = df[
			(df['host_is_superhost'].isin([0, 1]))
			& (df['instant_bookable'].isin([0, 1]))
			& (pd.to_numeric(df['latitude'], errors='coerce').notnull())
			& (pd.to_numeric(df['longitude'], errors='coerce').notnull())
			# & (df['property_type'].isin(valid_property_types))
			# & (df['room_type'].isin(valid_room_types))
			& (pd.to_numeric(df['accommodates'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['bathrooms'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['bedrooms'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['beds'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['review_scores_rating'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['minimum_nights'], errors='coerce').between(1, 365))
			& (pd.to_numeric(df['number_of_reviews'], errors='coerce') > 0)
			& (pd.to_numeric(df['price'], errors='coerce').between(15, 10000))
			& (pd.to_numeric(df['host_total_listings_count'], errors='coerce').between(0, 10000))
			# & (df['neighbourhood_cleansed'].isin(valid_neighbourhood_cleansed))
			& (pd.to_numeric(df['review_scores_accuracy'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_cleanliness'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_checkin'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_communication'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_location'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_value'], errors='coerce').between(0, 5))
			# & (pd.to_numeric(df['bedrooms_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['beds_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['bathrooms_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_rating_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_accuracy_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_cleanliness_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_checkin_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_communication_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_location_na'], errors='coerce').between(0, 2))
			# & (pd.to_numeric(df['review_scores_value_na'], errors='coerce').between(0, 2))
		]
		logging.info(f"clean_dataset beds = {clean_dataset['beds']}")
		logging.info(f"clean_dataset bedrooms= {clean_dataset['bedrooms']}")
		logging.info(f"clean_dataset bathrooms = {clean_dataset['bathrooms']}")
		logging.info(f"clean_dataset review_scores_rating = {clean_dataset['review_scores_rating']}")
		logging.info(f"clean_dataset review_scores_accuracy = {clean_dataset['review_scores_accuracy']}")
		logging.info(f"clean_dataset review_scores_cleanliness = {clean_dataset['review_scores_cleanliness']}")
		logging.info(f"clean_dataset review_scores_checkin = {clean_dataset['review_scores_checkin']}")
		logging.info(f"clean_dataset review_scores_communication = {clean_dataset['review_scores_communication']}")
		logging.info(f"clean_dataset review_scores_location = {clean_dataset['review_scores_location']}")
		logging.info(f"clean_dataset review_scores_value = {clean_dataset['review_scores_value']}")
		logging.info(f"clean_dataset review_scores_rating_na = {clean_dataset['review_scores_rating_na']}")
		logging.info(f"clean_dataset review_scores_accuracy_na = {clean_dataset['review_scores_accuracy_na']}")
		logging.info(f"clean_dataset review_scores_cleanliness_na = {clean_dataset['review_scores_cleanliness_na']}")
		logging.info(f"clean_dataset review_scores_checkin_na = {clean_dataset['review_scores_checkin_na']}")
		logging.info(f"clean_dataset review_scores_communication_na = {clean_dataset['review_scores_communication_na']}")
		logging.info(f"clean_dataset review_scores_location_na = {clean_dataset['review_scores_location_na']}")
		logging.info(f"clean_dataset review_scores_value_na = {clean_dataset['review_scores_value_na']}")
		logging.info(f"clean_dataset bedrooms_na = {clean_dataset['bedrooms_na']}")
		logging.info(f"clean_dataset beds_na = {clean_dataset['beds_na']}")
		logging.info(f"clean_dataset bathrooms_na = {clean_dataset['bathrooms_na']}")
		logging.info(f"clean_dataset price = {clean_dataset['price']}")
		logging.info(f"clean_dataset accommodates = {clean_dataset['accommodates']}")
		logging.info(f"clean_dataset number_of_reviews = {clean_dataset['number_of_reviews']}")
		logging.info(f"clean_dataset minimum_nights = {clean_dataset['minimum_nights']}")
		logging.info(f"clean_dataset host_total_listings_count = {clean_dataset['host_total_listings_count']}")
		logging.info(f"clean_dataset latitude = {clean_dataset['latitude']}")
		logging.info(f"clean_dataset longitude = {clean_dataset['longitude']}")
		# logging.info(f"clean_dataset room_type = {clean_dataset['room_type']}")
		# logging.info(f"clean_dataset property_type = {clean_dataset['property_type']}")
		logging.info(f"clean_dataset host_is_superhost = {clean_dataset['host_is_superhost']}")
		logging.info(f"clean_dataset instant_bookable = {clean_dataset['instant_bookable']}")
		logging.info('===============================')
		logging.info('---------------------------------------------------------------')

		# Return the cleaned dataset
		return clean_dataset


def open_and_parse_file(file):
		"""
		Open and parse the file.
		"""
		file_path = FILE_DIR + file + ".parquet"
		# pd.read_parquet('example_pa.parquet', engine='pyarrow')
		logging.info(f"file: {file_path}")
		
		# Declare clean_dataset
		clean_dataset = pd.DataFrame()
		# Read the content of the file
		try:
				pf = ParquetFile(file_path)
				df = pf.to_pandas()

				clean_dataset = clean_data(df)		
				
				return clean_dataset
		except Exception as e:
				logging.error(f"Error opening the file: {e}")
				return {"error": str(e)}
		