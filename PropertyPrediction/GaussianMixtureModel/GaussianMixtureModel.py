import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance
from joblib import dump, load
from rdkit import Chem
from rdkit.Chem import Descriptors

import os

RDS = True
CWD = os.getcwd()

# Load dataset
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/FinalDataset.csv')
else:
    descriptors_df = pd.read_csv('Datasets/FinalDataset.csv')

# Select SMILES strings and target variable
X_smiles = descriptors_df['smiles']
y = descriptors_df['WhackVisc']

# Function to convert SMILES to molecular descriptors
def smiles_to_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptors = []
    
    for mol in mols:
        if mol is not None:
            descriptor_values = [desc(mol) for desc in [d[1] for d in Descriptors.descList]]
        else:
            descriptor_values = [np.nan] * len(descriptor_names)
        descriptors.append(descriptor_values)
    
    return pd.DataFrame(descriptors, columns=descriptor_names)

# Convert SMILES to descriptors
X_descriptors = smiles_to_descriptors(X_smiles)

# Handle missing values by filling with the mean of the column
X_descriptors.fillna(X_descriptors.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_descriptors)

# Hyperparameter grid
param_grid = {
    'n_components': [1, 2, 3, 4, 5],  # Number of mixture components
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Covariance types
    'max_iter': [100, 200, 300],  # Maximum number of iterations
    'init_params': ['kmeans', 'random']  # Initialization methods
}

# Custom scorer for mean squared error
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Function to perform grid search with varying dataset sizes and cross-validation
def perform_grid_search(X, y, param_grid, dataset_sizes, k=5):
    results = []
    best_score = float('inf')
    best_params = None
    best_model = None
    total_models = 0
    
    for size in dataset_sizes:
        print(f"Dataset size: {size*100}%")
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=size, random_state=42)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(GaussianMixture(), param_grid, scoring=mse_scorer, cv=kf, n_jobs=-1)
        grid_search.fit(X_subset)
        
        best_estimator = grid_search.best_estimator_
        best_score_in_size = grid_search.best_score_
        best_params_in_size = grid_search.best_params_
        
        results.append({
            'params': best_params_in_size,
            'dataset_size': size,
            'avg_val_score': -best_score_in_size,  # convert back to positive
            'training_time': grid_search.refit_time_
        })
        
        total_models += len(grid_search.cv_results_['params'])
        
        if best_score_in_size < best_score:
            best_score = best_score_in_size
            best_params = best_params_in_size
            best_model = best_estimator

    return best_model, best_params, -best_score, results, total_models

# Define dataset sizes (20% to 100%)
dataset_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

# Perform grid search
best_model, best_params, best_score, results, total_models = perform_grid_search(X_scaled, y, param_grid, dataset_sizes)
print(f"Best hyperparameters: {best_params}, Best Score: {best_score}")
print(f"Total models trained: {total_models}")

# Save the best model
dump(best_model, 'best_gmm_model.joblib')

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('gmm_grid_search_results.csv', index=False)

# Plot performance
results_df.plot(x='dataset_size', y='avg_val_score', kind='scatter', title='Model Performance')
plt.xlabel('Dataset Size')
plt.ylabel('Average Validation MSE')
plt.show()

# Feature importance (permutation importance)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
best_model.fit(np.hstack((X_train, y_train.reshape(-1, 1))))
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_descriptors.columns)[sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance')
plt.show()

# Additional benchmarking measures
results_df['training_time'] = results_df['training_time'].astype(float)
avg_training_time = results_df['training_time'].mean()
print(f"Average training time per model: {avg_training_time} seconds")

# Best model evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error of the best model: {mse}')
