import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import xgboost as xgb
import os

RDS = False
CWD = os.getcwd()

# Load dataset
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/GridSearchDataset_Descriptors.csv')
    test_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/FinalTestDataset_Descriptors.csv')
else:
    descriptors_df = pd.read_csv('Datasets/GridSearchDataset_Descriptors.csv')
    test_df = pd.read_csv('Datasets/FinalTestDataset_Descriptors.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']
X_test = test_df.drop(columns=['Viscosity'])
y_test = test_df['Viscosity']

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000, 2000, 5000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
    'max_depth': [3, 4, 5, 6, 10, 20, 50, 100],
    'tree_method': ['hist'],  # Use GPU
}

# Custom grid search function
def custom_grid_search(X_train, y_train, X_test, y_test, param_grid):
    results = []
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'tree_method': 'hist',
                    'use_label_encoder': False,
                    'eval_metric': 'rmse'
                }
                model = xgb.XGBRegressor(**params)
                
                # Measure training time
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                train_time = end_time - start_time

                # Evaluate on test set
                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                result = {
                    'Params': params,
                    'Test MSE': test_mse,
                    'Test R2': test_r2,
                    'Train Time': train_time
                }
                results.append(result)

                # Save results to a CSV file in append mode
                results_df = pd.DataFrame([result])
                results_df.to_csv('xgboost_model_training_results.csv', mode='a', header=not os.path.isfile('xgboost_model_training_results.csv'), index=False)
                
                print(f"Params: {params}, Test MSE: {test_mse}, Test R2: {test_r2}")

    return results

# Perform custom grid search
results = custom_grid_search(X_train_scaled, y_train, X_test_scaled, y_test, param_grid)

# Find the best model based on test MSE
best_result = min(results, key=lambda x: x['Test MSE'])
best_params = best_result['Params']

# Train the best model on the full training data
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train_scaled, y_train)

# Save the best model
dump(best_model, 'best_xgboost_model.joblib')

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Best Params: {best_params}, Test MSE: {test_mse}, Test R2: {test_r2}")

# Plot performance of each model tested
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['Params']['n_estimators'], result['Test MSE'], 'o-', label=f"Learning Rate: {result['Params']['learning_rate']}, Max Depth: {result['Params']['max_depth']}")
plt.xlabel('Number of Estimators')
plt.ylabel('Test MSE')
plt.title('Model Performance During Custom Grid Search')
plt.legend()
plt.grid(True)
plt.savefig('xgboost_model_performance.png')
plt.show()

# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for XGBoost Regressor')
plt.savefig('feature_importance.png')
plt.show()
