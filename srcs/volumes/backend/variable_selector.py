import logging, requests, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

def select_features(df, target='price', threshold_target=0.1, threshold_features=0.3):
    """
    Select features based on their correlation with the target and remove highly correlated features.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target variable.
        target (str): The target variable for the model (default is 'price').
        threshold_target (float): Minimum absolute correlation with the target for a feature to be selected.
        threshold_features (float): Maximum allowed correlation between selected features.

    Returns:
        list: Selected feature names.
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    logging.info(f"correlation_matrix: {correlation_matrix}")

    # Extract correlations with the target variable
    correlation_with_target = correlation_matrix[target].sort_values(ascending=False)
    logging.info(f"correlation_with_target: {correlation_with_target}")
    
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

    # Select features with high correlation to the target
    potential_features = correlation_with_target[abs(correlation_with_target) > threshold_target].index.tolist()
    # Drop price from potential features
    potential_features.remove('price')

    logging.info(f"potential_features: {potential_features}")
    # Avoid autocorrelation between selected features
    selected_features = []
    for feature in potential_features:
        # Check correlation with already selected features
        if not selected_features:
            selected_features.append(feature)
        else:
            is_correlated = any(
                abs(correlation_matrix[feature][selected_feature]) > threshold_features
                for selected_feature in selected_features
            )
            if not is_correlated:
                selected_features.append(feature)
    logging.info(f"selected_features: {selected_features}")
    # Add price to selected features
    selected_features.append('price')
    return selected_features



def select_pertinent_data(data):
	# Select features
	selected_data = select_features(data)

	# Delete columns with low correlation
	for column in data.columns:
		if column not in selected_data:
			del data[column]

	return selected_data