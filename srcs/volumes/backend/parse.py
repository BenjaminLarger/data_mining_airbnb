import logging, requests, os
import pandas as pd
from fastparquet import ParquetFile

logging.basicConfig(level=logging.INFO)

FILE_DIR = "build/utils/subject/"

import pandas as pd

def extract_neighborhood_data(df):
    """
    Extracts neighborhood data as a list of tuples with the neighborhood name
    and its average price.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'neighbourhood_cleansed' and 'price' columns.

    Returns:
        list: A list of tuples where each tuple contains the neighborhood name and its average price.
    """
    # Ensure the 'price' column is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Group by 'neighbourhood_cleansed' and calculate the mean price
    grouped_data = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index()

    # Sort by price in descending order
    grouped_data = grouped_data.sort_values(by='price', ascending=False)

    # Convert the result to a list of tuples
    neighborhood_data = list(grouped_data.itertuples(index=False, name=None))

    return neighborhood_data

def extract_roomtype_data(df):
    """
    Extracts neighborhood data as a list of tuples with the neighborhood name
    and its average price.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'neighbourhood_cleansed' and 'price' columns.

    Returns:
        list: A list of tuples where each tuple contains the neighborhood name and its average price.
    """
    # Ensure the 'price' column is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Group by 'room_type' and calculate the mean price
    grouped_data = df.groupby('room_type')['price'].mean().reset_index()

    # Sort by price in descending order
    grouped_data = grouped_data.sort_values(by='price', ascending=False)

    # Convert the result to a list of tuples
    roomtype_data = list(grouped_data.itertuples(index=False, name=None))

    return roomtype_data

def extract_propertytype_data(df):
    """
    Extracts neighborhood data as a list of tuples with the neighborhood name
    and its average price.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'neighbourhood_cleansed' and 'price' columns.

    Returns:
        list: A list of tuples where each tuple contains the neighborhood name and its average price.
    """
    # Ensure the 'price' column is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Group by 'room_type' and calculate the mean price
    grouped_data = df.groupby('property_type')['price'].mean().reset_index()

    # Sort by price in descending order
    grouped_data = grouped_data.sort_values(by='price', ascending=False)

    # Convert the result to a list of tuples
    propertytype_data = list(grouped_data.itertuples(index=False, name=None))

    return propertytype_data

def generate_neighborhood_map(neighborhood_data):
    """
    Generate a mapping from neighborhood names to their ranking index.

    Returns:
        dict: A dictionary where keys are neighborhood names and values are their ranks.
    """

    # Sort neighborhoods by their values (descending order, assuming it's average price)
    sorted_neighborhoods = sorted(neighborhood_data, key=lambda x: x[1], reverse=False)

    # Generate the mapping
    neighborhood_map = {neighborhood: rank for rank, (neighborhood, _) in enumerate(sorted_neighborhoods, start=1)}

    return neighborhood_map

def generate_roomtype_map(roomtype_data):
    """
    Generate a mapping from neighborhood names to their ranking index.

    Returns:
        dict: A dictionary where keys are neighborhood names and values are their ranks.
    """

    # Sort neighborhoods by their values (descending order, assuming it's average price)
    sorted_roomtype = sorted(roomtype_data, key=lambda x: x[1], reverse=True)

    # Generate the mapping
    roomtype_map = {neighborhood: rank for rank, (neighborhood, _) in enumerate(sorted_roomtype, start=1)}

    return roomtype_map

def generate_propertytype_map(propertytype_data):
    """
    Generate a mapping from neighborhood names to their ranking index.

    Returns:
        dict: A dictionary where keys are neighborhood names and values are their ranks.
    """

    # Sort neighborhoods by their values (descending order, assuming it's average price)
    sorted_propertytype = sorted(propertytype_data, key=lambda x: x[1], reverse=True)

    # Generate the mapping
    propertytype_map = {neighborhood: rank for rank, (neighborhood, _) in enumerate(sorted_propertytype, start=1)}

    return propertytype_map


def get_average_price_per_neighborhood(df):
		"""
		Calculate the average price per neighborhood from a given DataFrame.

		Args:
		 df (pd.DataFrame): Input DataFrame with 'neighbourhood_cleansed' and 'price' columns.

		Returns:
		 pd.DataFrame: A DataFrame with neighborhoods and their average prices.
		"""
		# Ensure the 'price' column is numeric
		df['price'] = pd.to_numeric(df['price'], errors='coerce')

		# Group by 'neighbourhood_cleansed' and calculate the average price
		average_price_per_neighborhood = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index()

		# Rename the columns for better readability
		average_price_per_neighborhood.columns = ['Neighbourhood', 'Average Price']

		# Rank the neighborhoods by average price
		average_price_per_neighborhood = average_price_per_neighborhood.sort_values(by='Average Price', ascending=False)

		# Check the DataFrame after sorting
		logging.info(f"average_price_per_neighborhood after sorting: {average_price_per_neighborhood}")

		return average_price_per_neighborhood

# Example of usage
# result = get_average_price_per_neighborhood(your_dataframe)
# print(result)


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

		# Convert string columns to numeric
		neighborhood_data = extract_neighborhood_data(df)
		logging.info(f"neighborhood_data: {neighborhood_data}")
		neighborhood_map = generate_neighborhood_map(neighborhood_data)
		logging.info(f"neighborhood_map: {neighborhood_map}")
		df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].map(neighborhood_map)

		roomtype_data = extract_roomtype_data(df)
		logging.info(f"roomtype_data: {roomtype_data}")
		roomtype_map = generate_roomtype_map(roomtype_data)
		logging.info(f"roomtype_map: {roomtype_map}")
		df['room_type'] = df['room_type'].map(roomtype_map)

		propertytype_data = extract_propertytype_data(df)
		logging.info(f"propertytype_data: {propertytype_data}")
		propertytype_map = generate_propertytype_map(propertytype_data)
		logging.info(f"propertytype_map: {propertytype_map}")
		df['property_type'] = df['property_type'].map(propertytype_map)

		df['host_is_superhost'] = df['host_is_superhost'].map({'f': 0, 't': 1})
		df['instant_bookable'] = df['instant_bookable'].map({'f': 0, 't': 1})
		average_price_per_neighborhood = get_average_price_per_neighborhood(df)
		# Create a map of neighborhoods with host_is_superhost and instant_bookable
		neighborhood_map = df.groupby('neighbourhood_cleansed').agg({
			'host_is_superhost': 'mean',
			'instant_bookable': 'mean'
		}).reset_index()

		# Merge the average price per neighborhood with the neighborhood map
		average_price_per_neighborhood = average_price_per_neighborhood.merge(
			neighborhood_map, left_on='Neighbourhood', right_on='neighbourhood_cleansed', how='left'
		).drop(columns=['neighbourhood_cleansed'])

		# Rename the columns for better readability
		average_price_per_neighborhood.columns = [
			'Neighbourhood', 'Average Price', 'Superhost Ratio', 'Instant Bookable Ratio'
		]
		
		clean_dataset = df[
			(df['host_is_superhost'].isin([0, 1]))
			& (df['instant_bookable'].isin([0, 1]))
			& (pd.to_numeric(df['latitude'], errors='coerce').notnull())
			& (pd.to_numeric(df['longitude'], errors='coerce').notnull())
			& (pd.to_numeric(df['accommodates'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['bathrooms'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['bedrooms'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['beds'], errors='coerce').between(1, 100))
			& (pd.to_numeric(df['review_scores_rating'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['minimum_nights'], errors='coerce').between(1, 365))
			& (pd.to_numeric(df['number_of_reviews'], errors='coerce') >= 0)
			& (pd.to_numeric(df['price'], errors='coerce').between(15, 10000))
			& (pd.to_numeric(df['host_total_listings_count'], errors='coerce').between(0, 10000))
			& (pd.to_numeric(df['review_scores_accuracy'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_cleanliness'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_checkin'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_communication'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_location'], errors='coerce').between(0, 5))
			& (pd.to_numeric(df['review_scores_value'], errors='coerce').between(0, 5))
		]

		# Return the cleaned dataset
		clean_dataset = clean_dataset[[col for col in clean_dataset.columns if not col.endswith('_na')]]

		return clean_dataset


def open_and_parse_file(file):
		"""
		Open and parse the file.
		"""
		file_path = FILE_DIR + file + ".parquet"
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
		