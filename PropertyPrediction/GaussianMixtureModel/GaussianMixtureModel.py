import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import csv

RDS = False

# Load the dataset
if RDS:
    df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/GridSearchDataset.csv')
else:
    df = pd.read_csv('Datasets/GridSearchDataset.csv')

# Feature extraction function
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors._descList))  # Handle invalid SMILES
    return np.array([desc(mol) for name, desc in Descriptors._descList])

# Extract features and target variable
X = np.array([smiles_to_features(smiles) for smiles in df['smiles']])
y = df['visco@40C[cP]'].values

# Define parameter grid
param_grid = {
    'n_components': [1, 2, 3, 4, 5, 10, 15, 20],
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'max_iter': [100, 200, 300, 500, 1000],
    'init_params': ['kmeans', 'random'],
    'reg_covar': [1e-6, 1e-5, 1e-4]
}

# Custom grid search function
def custom_grid_search(X, y, param_grid, cv=5):
    results = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Prepare CSV file
    with open('gmm_grid_search_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n_components', 'covariance_type', 'max_iter', 'init_params', 'reg_covar', 'avg_train_time', 'avg_mse'])

        # Grid search
        for n_components in param_grid['n_components']:
            for covariance_type in param_grid['covariance_type']:
                for max_iter in param_grid['max_iter']:
                    for init_params in param_grid['init_params']:
                        for reg_covar in param_grid['reg_covar']:
                            mse_scores = []
                            train_times = []

                            print(n_components, covariance_type)
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]

                                # Train model
                                model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, init_params=init_params, reg_covar=reg_covar, random_state=42)
                                start_time = time.time()
                                model.fit(X_train)
                                train_time = time.time() - start_time

                                # Predict and evaluate
                                y_pred = model.predict(X_test)
                                mse = mean_squared_error(y_test, y_pred)

                                mse_scores.append(mse)
                                train_times.append(train_time)
                            
                            # Calculate average metrics
                            avg_mse = np.mean(mse_scores)
                            avg_train_time = np.mean(train_times)

                            # Save results
                            writer.writerow([n_components, covariance_type, max_iter, init_params, reg_covar, avg_train_time, avg_mse])
                            file.flush()
                            results.append((n_components, covariance_type, max_iter, init_params, reg_covar, avg_train_time, avg_mse))
    
    return results

# Perform grid search
results = custom_grid_search(X, y, param_grid)
