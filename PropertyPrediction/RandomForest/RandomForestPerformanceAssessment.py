import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the CSV files
grid_search_results = pd.read_csv('PropertyPrediction/RandomForest/random_forest_model_training_results.csv')
train_dataset = pd.read_csv('Datasets/LargeTrainingDataset_Descriptors.csv')
test_dataset = pd.read_csv('Datasets/FinalTestDataset_Descriptors.csv')

# Step 1: Identify the best performing model according to MSE
best_model_row = grid_search_results.loc[grid_search_results['Validation MSE'].idxmin()]
best_params = json.loads(best_model_row['Params'].replace("'", "\""))
print(best_params)

# Extract the features and target variable from the training and test datasets
X_train = train_dataset.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Viscosity'])
y_train = train_dataset['Viscosity']
X_test = test_dataset.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Viscosity'])
y_test = test_dataset['Viscosity']

# Step 2: Function to perform 5-fold cross-validation and collect MSE scores using the test dataset
def cross_val_mse(model, X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    return np.mean(mse_scores), np.std(mse_scores)

# Step 3: Perform 5-fold cross-validation on different sizes of the training dataset
dataset_sizes = [0.2, 0.4, 0.6, 0.8, 0.999]
results = []

for size in dataset_sizes:
    print(f'Training Size: {size}')
    X_subset = X_train.sample(frac=size, random_state=42)
    y_subset = y_train.loc[X_subset.index]
    model = RandomForestRegressor(**best_params)
    mse, std = cross_val_mse(model, X_subset, y_subset, X_test, y_test)
    results.append((size, mse, std))

# Step 4: Evaluate and Plot Performance
sizes, mses, stds = zip(*results)

plt.figure(figsize=(10, 6))
plt.errorbar(sizes, mses, yerr=stds, fmt='-o')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title('5-Fold Cross-Validation MSE vs Training Set Size')
plt.savefig('RF_Training.png')
plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=['Training Set Size', 'MSE', 'Standard Deviation'])
results_df.to_csv('RF_cross_validation_results.csv', index=False)

# Predict and time the predictions on the test set
start_time = time.time()
best_rf_model = RandomForestRegressor(**best_params)
best_rf_model.fit(X_train, y_train)
y_pred = best_rf_model.predict(X_test)
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
final_results_df.to_csv('final_results.csv', index=False)

print("Cross-validation results and final results have been saved to CSV files.")