# Weather Prediction Project

## Introduction  

This project predicts weather patterns using advanced machine learning models. By selecting a city, the program provides insights into temperature trends, precipitation levels, and other weather variables. The prediction system employs models like Random Forest, Gradient Boosting, Ridge Regression, and K-Nearest Neighbors, each evaluated based on their performance metrics.  

---

## Features  

- **Interactive Web Interface**: Access the application via a web browser to make predictions.  
- **Machine Learning Models**: Includes Random Forest, Gradient Boosting, Ridge Regression, and KNN.  
- **Real-time Compilation**: Automatically compiles and launches upon selection of a city.  
- **Performance Metrics**: View model evaluation metrics (MAE, RMSE, R²) within the interface.  

---

## Prerequisites  

Before launching the program, ensure you have:  
- A UNIX-based environment with `make` installed.  
- Docker installed
- Access to `localhost` on port `8443`.  

---

## How to Launch the Program  

1. Clone the repository and navigate to the root of the project directory.  
2. Build the program by running:  
   make
3. Open your browser and connect to: http://localhost:8443/
4. Select a city from the dropdown menu in the interface.
5. Wait for the system to complete the compilation and display results.


