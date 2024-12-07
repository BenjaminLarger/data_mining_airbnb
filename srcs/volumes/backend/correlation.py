import logging, requests, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def select_features(df):

	# Calculate the correlation matrix
	correlation_matrix = df.corr()
	logging.info(f"correlation_matrix: {correlation_matrix}")

	# Extract correlations with the target variable (price)
	correlation_with_target = correlation_matrix['price'].sort_values(ascending=False)
	logging.info(f"correlation_with_target: {correlation_with_target}")

	# Select features with high correlation to price
	selected_features = correlation_with_target[abs(correlation_with_target) > 0.1].index.tolist()
	return selected_features

def select_pertinent_data(data):
	# Select features
	selected_data = select_features(data)

	# Delete columns with low correlation
	for column in data.columns:
		if column not in selected_data:
			del data[column]

	return selected_data