from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def compare_models(random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results):
		# Extract random forest model metrics
		random_forest_mae_train = random_forest_results.get('MAE (Train)')
		random_forest_mae_test = random_forest_results.get('MAE (Test)')
		random_forest_rmse_train = random_forest_results.get('RMSE (Train)')
		random_forest_rmse_test = random_forest_results.get('RMSE (Test)')
		random_forest_r2_train = random_forest_results.get('R² (Train)')
		random_forest_r2_test = random_forest_results.get('R² (Test)')
		random_forest_cv_rmse = random_forest_results.get('Cross-Validation RMSE')

		# Extract gradient boosting model metrics
		gradient_boosting_mae_train = gradient_boosting_results.get('MAE (Train)')
		gradient_boosting_mae_test = gradient_boosting_results.get('MAE (Test)')
		gradient_boosting_rmse_train = gradient_boosting_results.get('RMSE (Train)')
		gradient_boosting_rmse_test = gradient_boosting_results.get('RMSE (Test)')
		gradient_boosting_r2_train = gradient_boosting_results.get('R² (Train)')
		gradient_boosting_r2_test = gradient_boosting_results.get('R² (Test)')
		gradient_boosting_cv_rmse = gradient_boosting_results.get('Cross-Validation RMSE')

		# Extract linear regression model metrics
		linear_regression_mae_train = linear_gradient_results.get('MAE (Train)')
		linear_regression_mae_test = linear_gradient_results.get('MAE (Test)')
		linear_regression_rmse_train = linear_gradient_results.get('RMSE (Train)')
		linear_regression_rmse_test = linear_gradient_results.get('RMSE (Test)')
		linear_regression_r2_train = linear_gradient_results.get('R² (Train)')
		linear_regression_r2_test = linear_gradient_results.get('R² (Test)')
		linear_regression_cv_rmse = linear_gradient_results.get('Cross-Validation RMSE')

		# Extract KNN model metrics
		knn_mae_train = knn_results.get('MAE (Train)')
		knn_mae_test = knn_results.get('MAE (Test)')
		knn_rmse_train = knn_results.get('RMSE (Train)')
		knn_rmse_test = knn_results.get('RMSE (Test)')
		knn_r2_train = knn_results.get('R² (Train)')
		knn_r2_test = knn_results.get('R² (Test)')
		knn_cv_rmse = knn_results.get('Cross-Validation RMSE')

		# Plotting
		labels = ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'KNN']
		train_mae = [random_forest_mae_train, gradient_boosting_mae_train, linear_regression_mae_train, knn_mae_train]
		test_mae = [random_forest_mae_test, gradient_boosting_mae_test, linear_regression_mae_test, knn_mae_test]
		train_rmse = [random_forest_rmse_train, gradient_boosting_rmse_train, linear_regression_rmse_train, knn_rmse_train]
		test_rmse = [random_forest_rmse_test, gradient_boosting_rmse_test, linear_regression_rmse_test, knn_rmse_test]
		train_r2 = [random_forest_r2_train, gradient_boosting_r2_train, linear_regression_r2_train, knn_r2_train]
		test_r2 = [random_forest_r2_test, gradient_boosting_r2_test, linear_regression_r2_test, knn_r2_test]
		cv_rmse = [random_forest_cv_rmse, gradient_boosting_cv_rmse, linear_regression_cv_rmse, knn_cv_rmse]

		x = np.arange(len(labels))
		width = 0.35

		fig, ax = plt.subplots(figsize=(12, 6))
		rects1 = ax.bar(x - width, train_mae, width, label='Train MAE')
		rects2 = ax.bar(x, test_mae, width, label='Test MAE')
		rects3 = ax.bar(x + width, train_rmse, width, label='Train RMSE')
		rects4 = ax.bar(x + 2 * width, test_rmse, width, label='Test RMSE')

		ax.set_ylabel('Scores')
		ax.set_title('Model Comparison')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()

		fig.tight_layout()

		plt.show()
		logging.info("Model comparison plot generated")



def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the accuracy and robustness of a machine learning model.

    Args:
        model: Trained model to evaluate.
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        Dict containing evaluation metrics.
    """
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Accuracy metrics on test data
    metrics = {
    	"MAE (Train)": mean_absolute_error(y_train, y_pred_train),
    	"MAE (Test)": mean_absolute_error(y_test, y_pred_test),
			"RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_pred_train)),
    	"RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    	"R² (Train)": r2_score(y_train, y_pred_train),
    	"R² (Test)": r2_score(y_test, y_pred_test)
    }

    # Robustness (Cross-Validation)
    cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    metrics["Cross-Validation RMSE"] = (-cross_val_scores.mean()) ** 0.5

    return metrics

# Example usage:
# metrics = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
# print(metrics)


def build_json_response(city, dictionary_response, selected_data, random_forest_results, gradient_boosting_results, linear_gradient_results, knn_results):
		random_forest_r2 = random_forest_results.get('R^2 Score')
		random_forest_rmse = random_forest_results.get('Root Mean Squared Error')
		gradient_boosting_r2 = gradient_boosting_results.get('R^2 Score')
		gradient_boosting_rmse = gradient_boosting_results.get('Root Mean Squared Error')
		linear_regression_r2 = linear_gradient_results.get('R-squared')
		linear_regression_rmse = linear_gradient_results.get('Root Mean Squared Error')
		knn_r2 = knn_results.get('R^2 Score')
		knn_rmse = knn_results.get('Root Mean Squared Error')
		return {
				'city': city,
				'latitude': dictionary_response.get('latitude'),
				'longitude': dictionary_response.get('longitude'),
				'data': selected_data,
				'random_forest_r2': random_forest_r2,
				'random_forest_rmse': random_forest_rmse,
				'gradient_boosting_r2': gradient_boosting_r2,
				'gradient_boosting_rmse': gradient_boosting_rmse,
				'linear_regression_r2': linear_regression_r2,
				'linear_regression_rmse': linear_regression_rmse,
				'knn_r2': knn_r2,
				'knn_rmse': knn_rmse
		}