import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer
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

# Define the scaler and model
scaler = StandardScaler()
model = SVR()

# Define the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'epsilon': [0.001, 0.01, 0.1, 0.2],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, verbose=2)

# Track time for training each model
start_time = time.time()

# Perform grid search and save results incrementally
results = []
for C in param_grid['C']:
    for epsilon in param_grid['epsilon']:
        for kernel in param_grid['kernel']:
            current_params = {'C': [C], 'epsilon': [epsilon], 'kernel': [kernel]}
            grid_search = GridSearchCV(estimator=model, param_grid=current_params, scoring='neg_mean_squared_error', cv=kf, verbose=2)
            grid_search.fit(scaler.fit_transform(X), y)

            # Save intermediate results
            for i in range(len(grid_search.cv_results_['mean_test_score'])):
                result = {
                    'params': grid_search.cv_results_['params'][i],
                    'mean_test_score': -grid_search.cv_results_['mean_test_score'][i],
                    'std_test_score': grid_search.cv_results_['std_test_score'][i],
                    'rank_test_score': grid_search.cv_results_['rank_test_score'][i]
                }
                results.append(result)
                with open(join(CWD, 'incremental_grid_search_results.csv'), mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if file.tell() == 0:
                        writer.writerow(['Parameters', 'Mean Test Score', 'Standard Deviation Test Score', 'Rank'])
                    writer.writerow([result['params'], result['mean_test_score'], result['std_test_score'], result['rank_test_score']])

# Calculate average time to train each model
end_time = time.time()
average_training_time = (end_time - start_time) / (kf.get_n_splits() * len(param_grid['C']) * len(param_grid['epsilon']) * len(param_grid['kernel']))

print(f"Average time to train each model: {average_training_time:.2f} seconds")

# Best model and parameters
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Save the best model
joblib.dump(best_model, join(CWD, 'best_svr_model.pkl'))

# Prepare to plot performance
results = grid_search.cv_results_
plt.figure(figsize=(12, 8))
plt.title("Grid Search Scores")

# Plot each model's performance
for i, param in enumerate(results['params']):
    plt.scatter(i, -results['mean_test_score'][i], label=str(param), marker='o')

plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.grid()
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Parameters")
plt.savefig(join(CWD, 'svr_model_performance.png'))
plt.show()

# Output final results to a CSV file
with open(join(CWD, 'grid_search_results.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Rank', 'Parameters', 'Mean Test Score', 'Standard Deviation Test Score'])
    for rank, params, mean_score, std_score in zip(results['rank_test_score'], results['params'], results['mean_test_score'], results['std_test_score']):
        writer.writerow([rank, params, -mean_score, std_score])

# Measure feature importance (coefficients for linear kernels, not available for other kernels)
if best_model.kernel == 'linear':
    feature_importance = np.abs(best_model.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(feature_importance)
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

# Example prediction
# smiles_example = "CCO"
# prediction = predict_from_smiles(smiles_example)
# print(f"Predicted property for {smiles_example}: {prediction[0]}")

# Additional measures for benchmarking
total_models_trained = len(param_grid['C']) * len(param_grid['epsilon']) * len(param_grid['kernel']) * kf.get_n_splits()
print(f"Total models trained: {total_models_trained}")

# Perform 5-fold cross-validation with varying dataset sizes
dataset_sizes = [0.2, 0.4]#, 0.6, 0.8]
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
