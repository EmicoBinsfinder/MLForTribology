import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import matplotlib.pyplot as plt
import joblib
import time
import csv
import os
from os.path import join

CWD = os.getcwd()

# Load dataset
dataset_path = 'Datasets/GridSearchDataset_Descriptors.csv'
descriptors_df = pd.read_csv(dataset_path)

# Remove Unnamed column
descriptors_df = descriptors_df.loc[:, ~descriptors_df.columns.str.contains('^Unnamed')]

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Define the scaler and model
scaler = StandardScaler()
model = SVR()

# Define the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 1, 3, 5, 10],
    'epsilon': [0.001, 0.01, 0.1, 0.2],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Track time for training each model
start_time = time.time()

# Perform grid search and save results incrementally
results = []
with open(join(CWD, 'incremental_grid_search_results.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Parameters', 'Mean Test Score', 'Standard Deviation Test Score', 'Rank', 'Training Time'])

    for C in param_grid['C']:
        for epsilon in param_grid['epsilon']:
            for kernel in param_grid['kernel']:
                current_params = {'C': [C], 'epsilon': [epsilon], 'kernel': [kernel]}
                grid_search = GridSearchCV(estimator=model, param_grid=current_params, scoring='neg_mean_squared_error', cv=kf, verbose=2)
                
                model_start_time = time.time()
                grid_search.fit(scaler.fit_transform(X), y)
                model_end_time = time.time()

                training_time = (model_end_time - model_start_time) / 5

                # Save intermediate results
                for i in range(len(grid_search.cv_results_['mean_test_score'])):
                    result = {
                        'params': grid_search.cv_results_['params'][i],
                        'mean_test_score': -grid_search.cv_results_['mean_test_score'][i],
                        'std_test_score': grid_search.cv_results_['std_test_score'][i],
                        'rank_test_score': grid_search.cv_results_['rank_test_score'][i],
                        'training_time': training_time
                    }
                    results.append(result)
                    writer.writerow([result['params'], result['mean_test_score'], result['std_test_score'], result['rank_test_score'], result['training_time']])
                    file.flush()  # Ensure the data is written to the file immediately

# Calculate average time to train each model
end_time = time.time()
average_training_time = (end_time - start_time) / (len(param_grid['C']) * len(param_grid['epsilon']) * len(param_grid['kernel']))

print(f"Average time to train each model: {average_training_time:.2f} seconds")

# Best model and parameters
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Save the best model
joblib.dump(best_model, join(CWD, 'best_svr_model.pkl'))

# Prepare to plot performance
results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 8))
plt.title("Grid Search Scores")

# Plot each model's performance
for i, param in enumerate(results_df['params']):
    plt.scatter(i, results_df['mean_test_score'][i], label=str(param), marker='o')

plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.grid()
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Parameters")
plt.savefig(join(CWD, 'svr_model_performance.png'))
plt.show()

# Output final results to a CSV file
results_df.to_csv(join(CWD, 'grid_search_results.csv'), index=False)

# Measure feature importance (coefficients for linear kernels, not available for other kernels)
if best_model.kernel == 'linear':
    feature_importance = np.abs(best_model.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print(feature_importance_df)
else:
    print("Feature importance not available for non-linear kernels.")

# Function to perform individual predictions based on SMILES strings
def predict_from_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    calculator = MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
    descriptors = calculator.CalcDescriptors(mol)
    scaled_descriptors = scaler.transform([descriptors])
    prediction = best_model.predict(scaled_descriptors)
    return prediction

# Additional measures for benchmarking
total_models_trained = len(param_grid['C']) * len(param_grid['epsilon']) * len(param_grid['kernel']) * kf.get_n_splits()
print(f"Total models trained: {total_models_trained}")

# Perform 5-fold cross-validation with varying dataset sizes
dataset_sizes = [0.99]  # Add other sizes if necessary
cv_scores = []

for size in dataset_sizes:
    subset_X, _, subset_y, _ = train_test_split(X, y, train_size=size, random_state=42)
    scores = cross_val_score(best_model, scaler.fit_transform(subset_X), subset_y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

# Plot cross-validation results for different dataset sizes
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, cv_scores, marker='o')
plt.title('5-Fold Cross-Validation Scores for Different Dataset Sizes')
plt.xlabel('Dataset Size')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.savefig(join(CWD, 'cv_scores_dataset_sizes.png'))

# Save cross-validation results to a CSV file
cv_results_df = pd.DataFrame({'Dataset Size': dataset_sizes, 'Mean Squared Error': cv_scores})
cv_results_df.to_csv(join(CWD, 'cv_results_dataset_sizes.csv'), index=False)