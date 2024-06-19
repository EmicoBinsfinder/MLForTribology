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

# Load dataset
Dataset = pd.read_csv('FinalDataset.csv')

# Preprocess data (assuming 'smiles' is already tokenized or use some features)
X = Dataset.drop(columns=['visco@40C[cP]', 'smiles'])
y = Dataset['visco@40C[cP]']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter grid
param_grid = {
    'n_components': [1, 2, 3, 4, 5],  # Number of mixture components
    'covariance_type': ['full', 'tied', 'diag', 'spherical']  # Covariance types
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
        
        for params in GridSearchCV(GaussianMixture(), param_grid, scoring=mse_scorer, cv=kf).get_params()['param_grid']:
            start_time = time.time()
            val_scores = []

            for train_index, val_index in kf.split(X_subset):
                X_train, X_val = X_subset[train_index], X_subset[val_index]
                y_train, y_val = y_subset[train_index], y_subset[val_index]
                
                # Combine features and target into one array for GMM
                train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
                val_data = np.hstack((X_val, y_val.reshape(-1, 1)))

                model = GaussianMixture(**params)
                model.fit(train_data)
                
                # Predict on validation set
                X_val_augmented = np.hstack((X_val, np.zeros((X_val.shape[0], 1))))
                responsibilities = model.predict_proba(X_val_augmented)
                means = model.means_[:, -1]
                y_val_pred = responsibilities @ means
                
                val_score = mean_squared_error(y_val, y_val_pred)
                val_scores.append(val_score)
            
            avg_val_score = np.mean(val_scores)
            training_time = time.time() - start_time
            results.append({
                'params': params,
                'dataset_size': size,
                'avg_val_score': avg_val_score,
                'training_time': training_time
            })
            total_models += 1
            
            if avg_val_score < best_score:
                best_score = avg_val_score
                best_params = params
                best_model = model

    return best_model, best_params, best_score, results, total_models

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
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
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
