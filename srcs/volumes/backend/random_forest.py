import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

# DESCRIPTION OF RANDOM FOREST MODEL
# Random Forest is an ensemble learning method that operates by constructing a multitude of 
# decision trees at training time and outputting the class that is the mode of the classes 
# (classification) or mean prediction (regression) of the individual trees.
# It is a bagging technique and is an improvement over a single decision tree.

def random_forest_model(columns, data):

	data_df = pd.DataFrame(data, columns=columns)  # Adjust column names as needed
	y = data_df['price']
	X = data_df.drop(columns=['price'])

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	param_grid = {
		'n_estimators': [50, 100, 200], # Number of trees in the forest
		'max_depth': [None, 10, 20], # Maximum depth of the tree
		'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
		'min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
		'max_features': ['auto', 'sqrt', 'log2'] # Number of features to consider when looking for the best split
	}

	grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
								 param_grid=param_grid,
								 cv=5,
								 n_jobs=-1,
								 verbose=2)

	grid_search.fit(X_train, y_train)

	# Best parameters from Grid Search
	best_params = grid_search.best_params_

	# Plot the grid search results
	results = grid_search.cv_results_
	logging.info(f"random forest Grid Search Results: {results}")
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.set_xlabel('Number of Estimators')
	ax.set_ylabel('Mean Test Score')
	ax.set_title('Grid Search Results')
	ax.grid(True)
	ax.plot(results['param_n_estimators'], -results['mean_test_score'], marker='o', label='Mean Test Score')
	ax.legend()

	# Train the model with the best parameters
	best_regressor = RandomForestRegressor(**best_params, random_state=42)
	best_regressor.fit(X_train, y_train)

	# Evaluate the model
	metrics = evaluate_model(best_regressor, X_train, y_train, X_test, y_test)
	logging.info(f"Random Forest Model Results: {metrics}")

	return metrics

def feature_importance_plot(model, X):
		importances = model.feature_importances_
		feature_names = X.columns

		# Plot feature importance
		plt.bar(feature_names, importances)
		plt.title("Feature Importance")
		plt.show()		