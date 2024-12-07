from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from utils import evaluate_model
logging.basicConfig(level=logging.INFO)

def linear_regression_model(columns, data):
    """
    Build and train a linear regression model to predict housing prices.

    Args:
        columns (list): List of significant variables for the model.
        data (pd.DataFrame): The dataset containing features and the target variable.

    Returns:
        model: The trained linear regression model.
        results (dict): A dictionary containing model evaluation metrics.
    """
    # Prepare the feature matrix (X) and target vector (y)
    X = data[columns]  # Use the specified columns as features
    logging.info(f"X: {X}")
    y = data['price']  # The target variable is the price of the housing unit
    logging.info(f"y: {y}")
    
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    rmse = mse ** 0.5  # Root Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # R-squared
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"Root Mean Squared Error: {rmse}")
    logging.info(f"R-squared: {r2}")

    # Store the results in a dictionary
    results = {
        'Root Mean Squared Error': rmse,
        'R-squared': r2
    }
    logging.info(f"Results: {results}")
    
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Return the trained model and evaluation results
    return metrics

# from sklearn.ensemble import RandomForestRegressor
	# results = {
  #       'Root Mean Squared Error': rmse,
  #       'R-squared': r2
  #   }

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

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

	# Extract the best parameters
	best_params = grid_search.best_params_
	best_score = -grid_search.best_score_  # Convert back to positive MSE

	# Train the model with the best parameters on the entire dataset
	best_ridge = Ridge(**best_params)
	best_ridge.fit(X, y)

	# Make predictions
	y_pred = best_ridge.predict(X)

	# Calculate evaluation metrics
	mse = mean_squared_error(y, y_pred)
	rmse = mse ** 0.5
	r2 = r2_score(y, y_pred)

	return {
		"best_params": best_params,
		"best_score": best_score,
		"Root Mean Squared Error": rmse,
		"R-squared": r2
	}

# Example Usage
# columns = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
# data = your_dataframe
# results = tune_ridge_model(columns, data)
# print("Best Parameters:", results['best_params'])
# print("Best RMSE:", results['best_score'] ** 0.5)

