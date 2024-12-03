from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import pandas as pd
import logging, requests, os
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def gradient_boosting_model(columns, data):
	logging.info("Gradient Boosting Model")
	# Create a DataFrame from the data
	data_df = pd.DataFrame(data, columns=columns)  # Adjust column names as needed
	logging.info(f"Data DF: {data_df}")
	y = data_df['price']
	X = data_df.drop(columns=['price'])

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Initialize the Gradient Boosting model
	gbr = GradientBoostingRegressor(
			n_estimators=100,       # Number of boosting stages
			learning_rate=0.1,      # Shrinks the contribution of each tree
			max_depth=3,            # Maximum depth of each tree
			random_state=42         # Ensures reproducibility
	)

	# Train the model
	gbr.fit(X_train, y_train)

	##########YOUTUBE TUTORIAL##########

	# Cross-Validation Score
	cross_val_score(gbr, X_train, y_train, cv=3, n_jobs=-1).mean()

	param_grid = {
			'n_estimators': [100, 200, 300],
			'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
			'max_depth': [3, 5, 7, 9],
	}

	# Grid Search
	gbr2 = GridSearchCV(gbr, param_grid, cv=3, n_jobs=-1)

	# Fit the model
	gbr2.fit(X_train, y_train)

	#logging.info("Best Parameters:", gbr2.best_params_)
	#logging.info("Best Score:", gbr2.best_score_)

	##########CHAT GPT##########
	# Make predictions
	# y_pred = gbr.predict(X_test)

	# # Calculate Mean Squared Error
	# mse = mean_squared_error(y_test, y_pred)
	# logging.info(f"Mean Squared Error: {mse}")

	# # Calculate R² Score
	# r2 = r2_score(y_test, y_pred)
	# logging.info(f"R² Score: {r2}")

	# # Feature Importance
	# feature_importances = gbr.feature_importances_
	# plt.barh(X.columns, feature_importances)
	# plt.xlabel("Feature Importance")
	# plt.ylabel("Features")
	# plt.title("Feature Importance in Gradient Boosting")
	# plt.show()

	# param_grid = {
	# 		'n_estimators': [100, 200, 300],
	# 		'learning_rate': [0.01, 0.1, 0.2],
	# 		'max_depth': [3, 5, 7],
	# }

	# grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
	# 													param_grid=param_grid, 
	# 													scoring='neg_mean_squared_error', 
	# 													cv=3)
	# grid_search.fit(X_train, y_train)

	# # Best Parameters
	# logging.info("Best Parameters:", grid_search.best_params_)
