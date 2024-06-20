import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

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

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt', 'log2']
}

# Perform manual grid search with cross-validation
def manual_grid_search(param_grid, X, y, train_sizes, k=5):
    keys, values = zip(*param_grid.items())
    best_score = float('inf')
    best_params = None
    header_written = False

    for size in train_sizes:
        X_train_partial, _, y_train_partial, _ = train_test_split(X, y, train_size=size, random_state=42)
        
        for v in product(*values):
            params = dict(zip(keys, v))
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            val_scores = []

            print(f"Training model with parameters: {params} and train size: {size}")

            for train_index, val_index in kf.split(X_train_partial):
                X_train_fold, X_val_fold = X_train_partial.iloc[train_index], X_train_partial.iloc[val_index]
                y_train_fold, y_val_fold = y_train_partial.iloc[train_index], y_train_partial.iloc[val_index]

                model = RandomForestRegressor(**params, random_state=42)
                model.fit(X_train_fold, y_train_fold)
                y_val_pred = model.predict(X_val_fold)
                val_score = mean_squared_error(y_val_fold, y_val_pred)
                val_scores.append(val_score)

            avg_val_score = np.mean(val_scores)
            print(f"Params: {params}, Train Size: {size}, Avg. Validation MSE: {avg_val_score}")

            # Save each model's performance to a CSV file
            result = {
                'Train Size': size,
                'Params': params,
                'Validation MSE': avg_val_score
            }
            results_df = pd.DataFrame([result])
            results_df.to_csv('random_forest_model_training_results.csv', mode='a', header=not header_written, index=False)
            header_written = True

            if avg_val_score < best_score:
                best_score = avg_val_score
                best_params = params

    return best_params

# Define train sizes
train_sizes = [0.2, 0.4, 0.6, 0.8]

# Perform grid search
best_params = manual_grid_search(param_grid, X, y, train_sizes)
print(f"Best hyperparameters: {best_params}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train final model with best hyperparameters
best_model = RandomForestRegressor(**best_params, random_state=42, verbose=2)
best_model.fit(X_train, y_train)

# Save the best model
joblib.dump(best_model, 'best_random_forest_model.pkl')

# Load the best model
loaded_model = joblib.load('best_random_forest_model.pkl')

# Make predictions with the best model
y_pred = loaded_model.predict(X_test)

# Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Cross-Validation Scores
cv_scores = cross_val_score(loaded_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-np.mean(cv_scores)}")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(loaded_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 5))

# Calculate mean and standard deviation
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure()
plt.title("Learning Curve for RandomForestRegressor")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.grid()

# Plot the mean train and test scores with standard deviation
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.savefig('training.png')
