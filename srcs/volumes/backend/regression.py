from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
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

    # Return the trained model and evaluation results
    return model, results

