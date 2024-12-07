from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from utils import evaluate_model
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
logging.basicConfig(level=logging.INFO)

# DESCRIPTION OF RIDGE REGRESSION MODEL
# Ridge Regression is a linear regression model that uses L2 regularization to prevent overfitting.
# It adds a penalty term to the loss function that is proportional to the square of the magnitude of the coefficients.
# This penalty term helps to reduce the complexity of the model and prevent overfitting.

def tune_ridge_model(columns, data):
	"""
	Tune hyperparameters for Ridge Regression.

	Args:
		columns (list): List of significant variables (features).
		data (pd.DataFrame): The dataset with features and target variable.

	Returns:
		dict: Best hyperparameters and corresponding performance metrics.
	"""
	# Features and target
	X = data[columns]
	y = data['price']

	# Define hyperparameter grid
	param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

	# Initialize the model
	ridge = Ridge()

	# Set up GridSearchCV
	grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

	# Perform the search
	grid_search.fit(X, y)
      
	# Plot the grid search results
	results = grid_search.cv_results_
	logging.info(f"Ridge Grid Search Results: {results}")
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.set_xlabel('Alpha')
	ax.set_ylabel('Mean Test Score')
	ax.set_title('Grid Search Results')
	ax.grid(True)
	ax.plot(param_grid['alpha'], -results['mean_test_score'], marker='o', label='Mean Test Score')
	ax.legend()

	# Extract the best parameters
	best_params = grid_search.best_params_
	best_score = -grid_search.best_score_  # Convert back to positive MSE

	# Train the model with the best parameters on the entire dataset
	best_ridge = Ridge(**best_params)
	best_ridge.fit(X, y)

	# Make predictions
	y_pred = best_ridge.predict(X)

	# Evaluate the model
	metrics = evaluate_model(best_ridge, X, y, X, y)
	logging.info(f"Ridge Metrics: {metrics}")

	return metrics


