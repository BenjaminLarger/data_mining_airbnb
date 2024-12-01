import logging, requests, os
import pandas as pd
from fastparquet import ParquetFile

logging.basicConfig(level=logging.INFO)

FILE_DIR = "build/utils/subject/"

def	clean_data(df):
	# clean_dataset init
	# Loop through df
    # if fd['host_is_superhost'] != f or t => continue
    # if fd['instant_bookable'] != f or t => continue
    # if fd['latitude'] is not a number => continue
    # if fd['longitude'] is not a number => continue
    # if fd['property_type'] is not 'Entire rental unit' or 'Private room in rental unit' or 'Room in hotel' or 'Entire home' or 'Private room in bed and breakfast' or 'Entire condo' => continue
    # if fd['room_type'] is not 'Private room' or 'Entire home/apt' or 'Hotel room' or 'Shared room' => continue
    # if fd['accomodates] is not a number greater than 0 and less than 100 => continue
    # if fd['bathrooms] is not a number greater than 0 and less than 100 => continue
    # if fd['bedrooms] is not a number greater than 0 and less than 100 => continue
    # if fd['beds] is not a number greater than 0 and less than 100 => continue
    # if fd['review_scores_rating] is not a number greater than 0 and less than 5 => continue
    # if fd['minimum_nights] is not a number greater than 0 and less than 365 => continue
    # if fd['number_of_reviews] is not a number greater than 0 => continue
    #return clean_dataset
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
    clean_dataset = df[
      (df['host_is_superhost'].isin(['f', 't'])) &
      (df['instant_bookable'].isin(['f', 't'])) &
      (pd.to_numeric(df['latitude'], errors='coerce').notnull()) &
      (pd.to_numeric(df['longitude'], errors='coerce').notnull()) &
      (df['property_type'].isin(valid_property_types)) &
      (df['room_type'].isin(valid_room_types)) &
      (pd.to_numeric(df['accommodates'], errors='coerce').between(1, 100)) &
      (pd.to_numeric(df['bathrooms'], errors='coerce').between(1, 100)) &
      (pd.to_numeric(df['bedrooms'], errors='coerce').between(1, 100)) &
      (pd.to_numeric(df['beds'], errors='coerce').between(1, 100)) &
      (pd.to_numeric(df['review_scores_rating'], errors='coerce').between(0, 5)) &
      (pd.to_numeric(df['minimum_nights'], errors='coerce').between(1, 365)) &
      (pd.to_numeric(df['number_of_reviews'], errors='coerce') > 0)
    	& (pd.to_numeric(df['price'], errors='coerce').between(15, 10000))
    	& (pd.to_numeric(df['host_total_listings_count'], errors='coerce').between(0, 10000))
    	# & (df['neighbourhood_cleansed'].isin(valid_neighbourhood_cleansed))
    	& (pd.to_numeric(df['review_scores_accuracy'], errors='coerce').between(0, 5))
    	& (pd.to_numeric(df['review_scores_cleanliness'], errors='coerce').between(0, 5))
    	& (pd.to_numeric(df['review_scores_checkin'], errors='coerce').between(0, 5))
    	& (pd.to_numeric(df['review_scores_communication'], errors='coerce').between(0, 5))
    	& (pd.to_numeric(df['review_scores_location'], errors='coerce').between(0, 5))
    	& (pd.to_numeric(df['review_scores_value'], errors='coerce').between(0, 5))
    	& (pd.to_numeric(df['bedrooms_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['beds_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['bathrooms_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_rating_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_accuracy_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_cleanliness_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_checkin_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_communication_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_location_na'], errors='coerce').between(0, 1))
    	& (pd.to_numeric(df['review_scores_value_na'], errors='coerce').between(0, 1))
    ]
    logging.info(f"Cleaned dataset: {clean_dataset}")
    logging.info(f"clean_dataset = {clean_dataset['review_scores_value_na']}")
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
        logging.info(f"df: {df}")
        # Print the price column
        logging.info(f"Price column: {df['price']} | instant bookable: {df['instant_bookable']} | latitude: {df['latitude']} | longitude: {df['longitude']} | property type: {df['property_type']} | room type: {df['room_type']} | accommodates: {df['accommodates']} | bathrooms: {df['bathrooms']} | bedrooms: {df['bedrooms']} | beds: {df['beds']} | reviews score rating: {df['review_scores_rating']} | minimum nights: {df['minimum_nights']}")

        clean_dataset = clean_data(df)
        
        # Write the cleaned dataset to a new csv file
        #clean_dataset.to_csv('cleaned_dataset.csv', index=False)        
        return clean_dataset
    except Exception as e:
        logging.error(f"Error opening the file: {e}")
        return {"error": str(e)}