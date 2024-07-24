import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from joblib import dump, load
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from itertools import product

# Function to convert SMILES to Morgan fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.zeros((n_bits,))

# Custom function to predict using GMM
def gmm_predict(estimator, X):
    responsibilities = estimator.predict_proba(X)
    means = estimator.means_[:, -1]
    y_pred = responsibilities @ means
    return y_pred

# Load dataset
RDS = False

if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/FinalDataset.csv')
else:
    descriptors_df = pd.read_csv('Datasets/FinalDataset.csv')

# Convert SMILES strings to numerical features
X = descriptors_df['smiles'].apply(smiles_to_fingerprint).tolist()
X = np.array(X)
y = descriptors_df['WhackVisc']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter grid
param_grid = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of mixture components
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Covariance types
    'max_iter': [100, 200, 300, 400, 500],  # Maximum number of iterations
    'init_params': ['kmeans', 'random'],  # Initialization methods
    'tol': [1e-4, 1e-3, 1e-2, 1e-1],  # Convergence threshold
    'reg_covar': [1e-6, 1e-5, 1e-4, 1e-3],  # Regularization term
}

# Function to perform grid search with varying dataset sizes and cross-validation
def perform_grid_search(X, y, param_grid, dataset_sizes, results_path='gmm_grid_search_results.csv', k=5):
    results = []
    best_score = float('inf')
    best_params = None
    best_model = None
    total_models = 0

    param_list = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    # Load previous results if they exist
    try:
        existing_results = pd.read_csv(results_path)
        results = existing_results.to_dict('records')
        total_models = len(results)
        best_score = min(result['avg_val_score'] for result in results)
    except FileNotFoundError:
        pass

    for size in dataset_sizes:
        print(f"Dataset size: {size*100}%")
        X_subset, _, y_subset, _ = train_test_split(X, y, train_size=size, random_state=42)
        y_subset = y_subset.values  # Convert to numpy array
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for params in param_list:
            param_dict = dict(zip(param_keys, params))
            model = GaussianMixture(**param_dict)

            cv_scores = []
            start_time = time.time()  # Moved start_time inside the loop to reset for each model
            for train_idx, val_idx in kf.split(X_subset):
                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = y_subset[train_idx], y_subset[val_idx]

                # Fit the model
                model.fit(X_train)

                # Predict using the model
                y_val_pred = gmm_predict(model, X_val)

                # Calculate the score
                score = mean_squared_error(y_val, y_val_pred)
                cv_scores.append(score)

            avg_val_score = np.mean(cv_scores)
            results.append({
                'params': param_dict,
                'dataset_size': size,
                'avg_val_score': avg_val_score,
                'training_time': time.time() - start_time
            })

            # Save the results incrementally
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_path, index=False)

            total_models += 1

            if avg_val_score < best_score:
                best_score = avg_val_score
                best_params = param_dict
                best_model = model

    return best_model, best_params, best_score, results, total_models

# Define dataset sizes (20% to 100%)
dataset_sizes = [0.99]#, 0.4, 0.6, 0.8, 1.0]

# Perform grid search
best_model, best_params, best_score, results, total_models = perform_grid_search(X_scaled, y, param_grid, dataset_sizes)
print(f"Best hyperparameters: {best_params}, Best Score: {best_score}")
print(f"Total models trained: {total_models}")

# Save the best model
dump(best_model, 'best_gmm_model.joblib')

# Load results for plotting
results_df = pd.read_csv('gmm_grid_search_results.csv')

# Plot performance
results_df.plot(x='dataset_size', y='avg_val_score', kind='scatter', title='Model Performance')
plt.xlabel('Dataset Size')
plt.ylabel('Average Validation MSE')
plt.show()

# Additional benchmarking measures
results_df['training_time'] = results_df['training_time'].astype(float)
avg_training_time = results_df['training_time'].mean()
print(f"Average training time per model: {avg_training_time} seconds")
