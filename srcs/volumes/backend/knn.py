from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utils import evaluate_model
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

# DESCRIPTION OF K-NEAREST NEIGHBORS MODEL
# K-Nearest Neighbors (KNN) is a simple, non-parametric, lazy learning algorithm 
# that can be used for both regression and classification tasks.
# It works by finding the 'k' nearest data points in the training set to a given test data point,
# and then predicting the target value based on the average (for regression) or majority vote (for classification) of the 'k' neighbors.

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

    # Plot Grid Search Results
    results = grid_search.cv_results_
    logging.info(f"KNN Grid Search Results: {results}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('Mean Test Score')
    ax.set_title('Grid Search Results')
    ax.grid(True)
    ax.plot(results['param_n_neighbors'], -results['mean_test_score'], marker='o', label='Mean Test Score')
    ax.legend()

    # Best model
    best_knn = grid_search.best_estimator_

    # Predictions
    y_pred = best_knn.predict(X_test)
   
    metrics = evaluate_model(best_knn, X_train, y_train, X_test, y_test)
    logging.info(f"KNN Metrics: {metrics}")
    return metrics

