import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load dataset
descriptors_df = pd.read_csv('Datasets/DecisionTreeDataset_313K.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform manual grid search with cross-validation
def manual_grid_search(param_grid, X, y, k=5):
    keys, values = zip(*param_grid.items())
    best_score = float('inf')
    best_params = None

    for v in product(*values):
        params = dict(zip(keys, v))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        val_scores = []

        print(f"Training model with parameters: {params}")

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, y_val_pred)
            val_scores.append(val_score)

        avg_val_score = np.mean(val_scores)
        print(f"Params: {params}, Avg. Validation MSE: {avg_val_score}")

        if avg_val_score < best_score:
            best_score = avg_val_score
            best_params = params

    return best_params

# Perform grid search
best_params = manual_grid_search(param_grid, X_train, y_train)
print(f"Best hyperparameters: {best_params}")

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
plt.show()
