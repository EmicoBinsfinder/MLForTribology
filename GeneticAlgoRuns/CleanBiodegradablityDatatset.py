import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import Descriptors

system = 'C:/Users/eeo21/VSCodeProjects/MLForTribology/PropertyPrediction/NISTMLModelTraining/GradientBoosting'

# Load dataset
file_path = f'{system}/BiodegradabilityDataset_Cleaned.csv'
df = pd.read_csv(file_path)

# Remove duplicates based on the SMILES column
df_cleaned = df.drop_duplicates(subset=['SMILES'])

# Generate molecular descriptors
descriptor_names = [desc[0] for desc in Descriptors.descList]
calc = MolecularDescriptorCalculator(descriptor_names)

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_names)
    return list(calc.CalcDescriptors(mol))

df_descriptors = df_cleaned['SMILES'].apply(compute_descriptors)
descriptor_df = pd.DataFrame(df_descriptors.tolist(), columns=descriptor_names)

# Merge descriptors with the target variable
df_final = pd.concat([descriptor_df, df_cleaned[['Activity']]], axis=1).dropna()

# Save descriptors to CSV
descriptor_csv_path = f'{system}/Biodegradability_Descriptors.csv'
df_final.to_csv(descriptor_csv_path, index=False)
print(f"Descriptors saved to: {descriptor_csv_path}")

# Separate features and target variable
X = df_final.drop(columns=['Activity'])
y = df_final['Activity']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000, 2000, 5000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
    'max_depth': [3, 4, 5, 6, 10, 20, 50, 100],
    'tree_method': ['hist'],  # Use GPU
}

# Custom grid search function
def custom_grid_search(X_train, y_train, X_test, y_test, param_grid):
    results = []
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'tree_method': 'hist',
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)

                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                train_time = end_time - start_time

                y_test_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred)

                result = {
                    'Params': params,
                    'Test Accuracy': test_accuracy,
                    'Test F1 Score': test_f1,
                    'Train Time': train_time
                }
                results.append(result)

                results_df = pd.DataFrame([result])
                results_df.to_csv(f'{system}/xgboost_Biodegradability_training_results.csv', mode='a', header=not os.path.isfile(f'{system}/xgboost_Biodegradability_training_results.csv'), index=False)
                
                print(f"Params: {params}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}")
    return results

# Perform custom grid search
results = custom_grid_search(X_train_scaled, y_train, X_test_scaled, y_test, param_grid)

# Find the best model based on test accuracy
best_result = max(results, key=lambda x: x['Test Accuracy'])
best_params = best_result['Params']

# Train the best model on the full training data
best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train_scaled, y_train)

# Save the best model
dump(best_model, f'{system}/best_xgboost_model_Biodegradability.joblib')

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
print(f"Best Params: {best_params}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}")

# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Biodegradability')
plt.savefig(f'{system}/feature_importance_Biodegradability.png')
plt.show()
