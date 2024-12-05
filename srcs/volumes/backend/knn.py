from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def knn_model(columns, data):
    """
    Build and train a K-Nearest Neighbors regression model for housing price prediction.

    Args:
        columns (list): Significant variables (features) for the model.
        data (pd.DataFrame): Parsed table containing features and target.

    Returns:
        Trained model and evaluation metrics.
    """
    # Features and target
    X = data[columns]
    y = data['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define KNN model
    knn = KNeighborsRegressor()

    # Grid Search for Hyperparameter Tuning
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model
    best_knn = grid_search.best_estimator_

    # Predictions
    y_pred = best_knn.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)


    results = {
				'R^2 Score': r2,
				'Root Mean Squared Error': rmse
		}
    
    return results

# Example usage:
# knn_model(['feature1', 'feature2', 'feature3'], df)
