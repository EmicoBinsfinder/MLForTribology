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


# Load dataset
descriptors_df = pd.read_csv('Datasets/DecisionTreeDataset_313K.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model and hyperparameters for grid search
model = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize lists to store results
results = []

# Perform 5-fold cross-validation with varying dataset sizes and grid search
train_sizes = [0.2, 0.4, 0.6, 0.8]
for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train_scaled, y_train, train_size=size, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    
    # Measure training time
    start_time = time.time()
    grid_search.fit(X_partial, y_partial)
    end_time = time.time()
    avg_train_time = (end_time - start_time) / (len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['p']) * len(param_grid['algorithm']))

    # Store results
    best_model = grid_search.best_estimator_
    cv_score = -grid_search.best_score_
    results.append({
        'Train Size': size,
        'Best Params': grid_search.best_params_,
        'CV Score': cv_score,
        'Train Time': avg_train_time
    })

    # Predict on the test set and evaluate
    y_pred = best_model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    results[-1]['Test MSE'] = test_mse
    results[-1]['Test R2'] = test_r2

    print(f"Training size: {size}, Best Params: {grid_search.best_params_}, CV MSE: {cv_score}, Test MSE: {test_mse}, Test R2: {test_r2}, Train Time: {avg_train_time}")

# Save the best model
best_model_idx = np.argmin([result['CV Score'] for result in results])
best_model = results[best_model_idx]['Best Params']
dump(best_model, 'best_knn_model.joblib')

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('knn_model_training_results.csv', index=False)

# Plot performance of each model tested
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['Train Size'], result['CV Score'], 'o-', label=f"Params: {result['Best Params']}")
plt.xlabel('Training Set Size')
plt.ylabel('Cross-Validation MSE')
plt.title('Model Performance During Grid Search and Cross-Validation')
plt.legend()
plt.grid(True)
plt.savefig('knn_model_performance.png')
plt.show()

# Plot the predicted vs actual values for the best model
best_model = KNeighborsRegressor(**best_model)
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)
plt.scatter(y_test, y_pred_best)
plt.xlabel('Actual Viscosity')
plt.ylabel('Predicted Viscosity')
plt.title('Actual vs Predicted Viscosity for Best KNN Model')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.savefig('knn_best_model_performance.png')
plt.show()