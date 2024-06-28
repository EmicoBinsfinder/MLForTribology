import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import os
from os.path import join 
import sys

RDS = False
CWD = os.getcwd()

# Load dataset
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/DecisionTreeDataset_313K.csv')
else:
    descriptors_df = pd.read_csv('Datasets/DecisionTreeDataset_313K.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model and hyperparameters for grid search
model = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 9, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski'],
    'p': [1, 2, 3],  # Only relevant if metric='minkowski'
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize lists to store results
results = []

# Perform 5-fold cross-validation with varying dataset sizes and grid search
train_sizes = [0.8, 0.6, 0.4, 0.2]
for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train_scaled, y_train, train_size=size, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    
    # Measure training time
    start_time = time.time()
    grid_search.fit(X_partial, y_partial)
    end_time = time.time()
    avg_train_time = (end_time - start_time) / (len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['p']) * len(param_grid['algorithm']))

    # Collect and save results for each parameter combination
    for i in range(len(grid_search.cv_results_['params'])):
        result = {
            'Train Size': size,
            'Params': grid_search.cv_results_['params'][i],
            'Mean Train Score': -grid_search.cv_results_['mean_train_score'][i],
            'Mean Test Score': -grid_search.cv_results_['mean_test_score'][i],
            'Std Test Score': grid_search.cv_results_['std_test_score'][i],
            'Train Time': avg_train_time
        }
        
        # Save results to a CSV file in append mode
        results_df = pd.DataFrame([result])
        results_df.to_csv(join(CWD, 'knn_model_training_results.csv'), mode='a', header=not os.path.isfile(join(CWD, 'knn_model_training_results.csv')), index=False)

    # Store best results
    best_model = grid_search.best_estimator_
    cv_score = -grid_search.best_score_

    # Predict on the test set and evaluate
    y_pred = best_model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"Training size: {size}, Best Params: {grid_search.best_params_}, CV MSE: {cv_score}, Test MSE: {test_mse}, Test R2: {test_r2}, Train Time: {avg_train_time}")

# Save the best model
best_model_idx = np.argmin([result['Mean Test Score'] for result in results])
best_model = grid_search.best_estimator_
dump(best_model, join(CWD, 'best_knn_model.joblib'))

# Plot performance of each model tested
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['Train Size'], result['Mean Test Score'], 'o-', label=f"Params: {result['Params']}")
plt.xlabel('Training Set Size')
plt.ylabel('Cross-Validation MSE')
plt.title('Model Performance During Grid Search and Cross-Validation')
plt.legend()
plt.grid(True)
plt.savefig(join(CWD, 'knn_model_performance.png'))

# Plot the predicted vs actual values for the best model
y_pred_best = best_model.predict(X_test_scaled)
plt.scatter(y_test, y_pred_best)
plt.xlabel('Actual Viscosity')
plt.ylabel('Predicted Viscosity')
plt.title('Actual vs Predicted Viscosity for Best KNN Model')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.savefig(join(CWD, 'knn_best_model_performance.png'))
