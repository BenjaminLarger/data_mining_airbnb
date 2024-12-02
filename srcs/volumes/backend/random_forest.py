import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
logging.basicConfig(level=logging.INFO)

def random_forest_model(columns, data):
	logging.info("Random Forest Model")
	logging.info(f"Columns: {columns}")


	data_df = pd.DataFrame(data, columns=columns)  # Adjust column names as needed
	logging.info(f"Data DF: {data_df}")
	y = data_df['price']
	X = data_df.drop(columns=['price'])

	# Split the data into training and testing sets
	logging.info(f"X: {X}")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	logging.info(f"X_train: {X_train}")
	logging.info(f"X_test: {X_test}")
	logging.info(f"y_train: {y_train}")
	logging.info(f"y_test: {y_test}")

	# Train a Random Forest Regressor
	regressor = RandomForestRegressor(n_estimators=100, random_state=42)
	regressor.fit(X_train, y_train)

	# Evaluate the model
	y_pred = regressor.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	logging.info(f"Root Mean Squared Error: {rmse}")

	r2_score = regressor.score(X_test, y_test)
	logging.info(f"R^2 Score: {r2_score}")

	# Feature importance
	feature_importance_plot(regressor, X)

	logging.info("-----------------")
	return regressor

# from sklearn.model_selection import GridSearchCV

# # Define parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Grid search with cross-validation
# grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
#                            param_grid=param_grid,
#                            cv=5,
#                            n_jobs=-1, 
#                            verbose=2)

# grid_search.fit(X_train, y_train)

# # Best parameters
# print(f"Best Parameters: {grid_search.best_params_}")

def feature_importance_plot(model, X):
		importances = model.feature_importances_
		feature_names = X.columns

		# Plot feature importance
		plt.bar(feature_names, importances)
		plt.title("Feature Importance")
		plt.show()		