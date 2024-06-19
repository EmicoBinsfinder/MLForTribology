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

# Load in dataset
descriptors_df = pd.read_csv('Datasets/DecisionTreeDataset_313K.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Define the scaler and model
scaler = StandardScaler()
model = SVR()

# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, verbose=1)

# Track time for training each model
start_time = time.time()

# Perform grid search
grid_search.fit(scaler.fit_transform(X), y)

# Calculate average time to train each model
end_time = time.time()
average_training_time = (end_time - start_time) / (kf.get_n_splits() * len(param_grid['C']) * len(param_grid['epsilon']) * len(param_grid['kernel']))

print(f"Average time to train each model: {average_training_time:.2f} seconds")

# Best model and parameters
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Save the best model
joblib.dump(best_model, 'best_svr_model.pkl')

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
plt.show()

# Output results to a CSV file
with open('grid_search_results.csv', mode='w', newline='') as file:
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
smiles_example = "CCO"
prediction = predict_from_smiles(smiles_example)
print(f"Predicted property for {smiles_example}: {prediction[0]}")

# Additional measures for benchmarking
total_models_trained = len(param_grid['C']) * len(param_grid['epsilon']) * len(param_grid['kernel']) * kf.get_n_splits()
print(f"Total models trained: {total_models_trained}")
