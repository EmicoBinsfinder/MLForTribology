import pandas as pd
import json
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time
import ast
import sys

# Assuming the file has been correctly loaded
file_path = 'PropertyPrediction/GaussianMixtureModel/gmm_grid_search_results.csv'

# Load the CSV file
gmm_grid_search_results = pd.read_csv(file_path)


# Step 1: Identify the best performing model according to MSE
best_model_row = gmm_grid_search_results.loc[gmm_grid_search_results['avg_mse'].idxmin()].to_dict()

del best_model_row['avg_mse']
del best_model_row['avg_train_time']

best_params = best_model_row

# Extract the features and target variable from the training and test datasets
train_dataset = pd.read_csv('Datasets/Kajita_Dataset_Descriptors_Bare.csv')
test_dataset = pd.read_csv('Datasets/Experimental_Test_Dataset_Descriptors_Bare.csv')

X_train = train_dataset.drop(columns=['visco@40C[cP]', 'visco@100C[cP]'])
y_train = train_dataset['visco@100C[cP]']
X_test = test_dataset.drop(columns=['Experimental_40C_Viscosity', 'Experimental_100C_Viscosity'])
y_test = test_dataset['Experimental_100C_Viscosity']

# Step 2: Function to perform 5-fold cross-validation and collect MSE scores using the test dataset
def cross_val_mse(model, X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_train_fold)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    return np.mean(mse_scores), np.std(mse_scores)

# Step 3: Perform 5-fold cross-validation on different sizes of the training dataset
dataset_sizes = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.999]
results = []

for size in dataset_sizes:
    print(f'Training Size: {size}')
    X_subset = X_train.sample(frac=size, random_state=42)
    y_subset = y_train.loc[X_subset.index]
    model = GaussianMixture(**best_params)
    mse, std = cross_val_mse(model, X_subset, y_subset, X_test, y_test)
    results.append((size, mse, std))

# Step 4: Evaluate and Plot Performance
sizes, mses, stds = zip(*results)

plt.errorbar(sizes, mses, yerr=stds, fmt='-o')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title('5-Fold Cross-Validation MSE vs Training Set Size (GMM)')
plt.savefig('gmm_training.png')
plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=['Training Set Size', 'MSE', 'Standard Deviation'])
results_df.to_csv('cross_validation_results_gmm.csv', index=False)

# Predict and time the predictions on the test set
start_time = time.time()
best_gmm_model = GaussianMixture(**best_params)
best_gmm_model.fit(X_train)
y_pred = best_gmm_model.predict(X_test)
prediction_time = time.time() - start_time
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)

# Save the final results
final_results = {
    'Time to Predict (per molecule)': prediction_time / len(y_test),
    'Test RMSE': rmse_test
}
final_results.update({f'MSE {int(size*100)}%': mse for size, mse, _ in results})
final_results.update({f'Std {int(size*100)}%': std for size, _, std in results})

final_results_df = pd.DataFrame([final_results])
final_results_df.to_csv('final_results_gmm.csv', index=False)

print("Cross-validation results and final results have been saved to CSV files.")
