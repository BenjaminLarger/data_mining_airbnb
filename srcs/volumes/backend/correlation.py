import logging, requests, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def select_features(df):
	logging.info("select_features")
	# Calculate the correlation matrix
	correlation_matrix = df.corr()
	logging.info(f"correlation_matrix: {correlation_matrix}")

	# Extract correlations with the target variable (price)
	correlation_with_target = correlation_matrix['price'].sort_values(ascending=False)
	logging.info(f"correlation_with_target: {correlation_with_target}")

	# Visualize correlations (optional)
	sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
	logging.info("Correlation Matrix Heatmap")
	plt.show()

	# Select features with high correlation to price
	selected_features = correlation_with_target[abs(correlation_with_target) > 0.1].index.tolist()
	print("Selected Features:", selected_features)
	return selected_features

def select_pertinent_data(data):

	# Delete valid_neighbourhood_cleansed column from the dataset
	del data['neighbourhood_cleansed']
	del data['property_type']
	del data['room_type']

	# Select features
	selected_data = select_features(data)


	return selected_data