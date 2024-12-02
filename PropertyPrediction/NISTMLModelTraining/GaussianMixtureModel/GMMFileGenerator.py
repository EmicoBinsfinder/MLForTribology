import os

# Define properties and corresponding filenames
properties = [
    ("Thermal_Conductivity_40C", "Thermal Conductivity at 40C"),
    ("Thermal_Conductivity_100C", "Thermal Conductivity at 100C"),
    ("Density_40C", "Density at 40C"),
    ("Density_100C", "Density at 100C"),
    ("Viscosity_40C", "Viscosity at 40C"),
    ("Viscosity_100C", "Viscosity at 100C"),
    ("Heat_Capacity_40C", "Heat Capacity at 40C"),
    ("Heat_Capacity_100C", "Heat Capacity at 100C")
]

# Template for the training script
script_template = """
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import csv

# Set the property name
property_name = "{property_name}"

# Load dataset for {description}
RDS = True
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Train_Descriptors.csv')
    test_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/LandoltNISTDatasets/{property_name}_Test_Descriptors.csv')
else:
    descriptors_df = pd.read_csv('Datasets/{property_name}_Descriptors.csv')
    test_df = pd.read_csv('Datasets/{property_name}_Test_Descriptors.csv')

# Feature extraction function
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors._descList))  # Handle invalid SMILES
    return np.array([desc(mol) for name, desc in Descriptors._descList])

# Extract features and target variable for training and testing
X = np.array([smiles_to_features(smiles) for smiles in descriptors_df['SMILES']])
y = descriptors_df[property_name].values

X_test = np.array([smiles_to_features(smiles) for smiles in test_df['SMILES']])
y_test = test_df[property_name].values

# Define parameter grid
param_grid = {{
    'n_components': [1, 2, 3, 4, 5],
    'covariance_type': ['full', 'tied', 'diag'],
    'max_iter': [100, 200, 300],
    'init_params': ['kmeans', 'random'],
    'reg_covar': [1e-6, 1e-5, 1e-4]
}}

# Custom grid search function
def custom_grid_search(X, y, param_grid, cv=5):
    results = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Prepare CSV file
    output_file = f'gmm_grid_search_results_{property_name}.csv'
    with open(output_file, mode='w', newline='') as file:
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

                            print(f"Training with parameters: {{n_components}}, {{covariance_type}}, {{max_iter}}, {{init_params}}, {{reg_covar}}")
                            for train_index, test_index in kf.split(X):
                                X_train, X_val = X[train_index], X[test_index]
                                y_train, y_val = y[train_index], y[test_index]

                                # Train model
                                model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, init_params=init_params, reg_covar=reg_covar, random_state=42)
                                start_time = time.time()
                                model.fit(X_train)
                                train_time = time.time() - start_time

                                # Predict and evaluate
                                y_pred = model.predict(X_val)
                                mse = mean_squared_error(y_val, y_pred)

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

# Evaluate the best model on the test set
best_params = min(results, key=lambda x: x[-1])  # Get the parameters with the lowest average MSE
n_components, covariance_type, max_iter, init_params, reg_covar, _, _ = best_params

print(f"Best parameters: {{best_params}}")

# Train final model on the entire dataset
final_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, init_params=init_params, reg_covar=reg_covar, random_state=42)
final_model.fit(X)

# Test the model on the test set
y_pred_test = final_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE: {{test_mse}}")
"""

# Generate individual scripts
output_dir = "GMM_Training_Scripts"
os.makedirs(output_dir, exist_ok=True)

for property_name, description in properties:
    script_path = os.path.join(output_dir, f"train_{property_name}.py")
    with open(script_path, "w") as f:
        f.write(script_template.format(property_name=property_name, description=description))
    print(f"Generated script: {script_path}")
