from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
import logging, requests, os
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate_model

logging.basicConfig(level=logging.INFO)

def gradient_boosting_model(columns, data):
	logging.info("Gradient Boosting Model")
	# Create a DataFrame from the data
	data_df = pd.DataFrame(data, columns=columns)
	logging.info(f"Data DF: {data_df.head()}")
	y = data_df['price']
	X = data_df.drop(columns=['price'])

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Initialize the Gradient Boosting model
	gbr = GradientBoostingRegressor(random_state=42)

	# Define the parameter grid for Grid Search
	param_grid = {
		'n_estimators': [100, 200, 300],
		'learning_rate': [0.01, 0.1, 0.2],
		'max_depth': [3, 5, 7]
	}

	# Perform Grid Search with cross-validation
	grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
	grid_search.fit(X_train, y_train)

	# Best Parameters
	logging.info(f"Best Parameters: {grid_search.best_params_}")
	logging.info(f"Best Score: {grid_search.best_score_}")

	# Make predictions with the best model
	best_model = grid_search.best_estimator_
	y_pred = best_model.predict(X_test)

	# Calculate Mean Squared Error
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)

	r2_score_best = best_model.score(X_test, y_test)
	best_params = grid_search.best_params_
	logging.info(f"Mean Squared Error: {mse}")

	results = {
		'R^2 Score': r2_score_best,
		'Root Mean Squared Error': rmse
	}

	metrics = evaluate_model(best_model, X_train, y_train, X_test, y_test)

	return metrics

	