import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
import time
import matplotlib.pyplot as plt
import ast

# Load datasets
training_data = pd.read_csv('Datasets/LargeTrainingDataset_Descriptors.csv')
test_data = pd.read_csv('Datasets/FinalTestDataset_Descriptors.csv')
grid_search_results = pd.read_csv('PropertyPrediction/KNN/grid_search_results.csv')

# Display the first few rows of each dataframe to understand their structure
print("Training Data:")
print(training_data.head())

print("\nTest Data:")
print(test_data.head())

print("\nGrid Search Results:")
print(grid_search_results.head())

# Assuming the target column is the last one in the datasets
target_column = training_data.columns[-1]
X_train = training_data.drop(columns=[target_column])
y_train = training_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# Finding the best performing model based on MSE from the grid search results
best_model_row = grid_search_results.loc[grid_search_results['mean_test_score'].idxmin()]
best_params = ast.literal_eval(best_model_row['params'])

print("\nBest Model Parameters based on Grid Search:")
print(best_params)

# Function to perform cross-validation and return the RMSE and standard deviation
def cross_val_performance(model, X_train, y_train, X_test, y_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for train_index, val_index in kf.split(X_train):
        X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_t, y_t)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        rmse_scores.append(rmse)
    return np.mean(rmse_scores), np.std(rmse_scores)

# Evaluating the best model on the test set
best_knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], 
                               weights=best_params['weights'], 
                               leaf_size=best_params['leaf_size'], 
                               p=best_params['p'], 
                               algorithm=best_params['algorithm'])

# Cross-validation with different sizes of the training dataset
dataset_sizes = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.999]
performance_results = []

for size in dataset_sizes:
    subset_X_train, _, subset_y_train, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    mean_rmse, std_rmse = cross_val_performance(best_knn, subset_X_train, subset_y_train, X_test, y_test)
    performance_results.append((size, mean_rmse, std_rmse))

# Plotting the performance results
sizes = [size * 100 for size in dataset_sizes]
mean_rmses = [result[1] for result in performance_results]
std_rmses = [result[2] for result in performance_results]

plt.figure(figsize=(10, 6))
plt.errorbar(sizes, mean_rmses, yerr=std_rmses, fmt='-o', capsize=5)
plt.xlabel('Training Dataset Size (%)')
plt.ylabel('RMSE')
plt.title('Test Set Performance at Different Training Data Sizes')
plt.grid(True)
plt.savefig('knn_test_performance_plot.png')
plt.show()

# Saving the results to a CSV file
results_df = pd.DataFrame({
    'Dataset Size (%)': sizes,
    'Mean RMSE': mean_rmses,
    'Std RMSE': std_rmses
})

results_df.to_csv('knn_test_performance_results.csv', index=False)

print("\nResults saved to 'knn_test_performance_results.csv' and 'knn_test_performance_plot.png'")
