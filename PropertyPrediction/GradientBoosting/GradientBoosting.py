import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt
import xgboost as xgb
import os

RDS = True
CWD = os.getcwd()

# Load dataset
if RDS:
    descriptors_df = pd.read_csv('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/Datasets/DecisionTreeDataset_313K.csv')
else:
    descriptors_df = pd.read_csv('Datasets/DecisionTreeDataset_313K.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'tree_method': ['hist'],  # Use GPU
    'device': ['cuda']
}

# Initialize lists to store results
results = []

# Perform 5-fold cross-validation with varying dataset sizes and grid search
train_sizes = [0.2, 0.4, 0.6, 0.8]
for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train_scaled, y_train, train_size=size, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse'),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Measure training time
    start_time = time.time()
    grid_search.fit(X_partial, y_partial)
    end_time = time.time()
    avg_train_time = (end_time - start_time) / (len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth']))

    # Collect and save results for each parameter combination
    for i in range(len(grid_search.cv_results_['params'])):
        result = {
            'Train Size': size,
            'Params': grid_search.cv_results_['params'][i],
            'Mean Train Score': -grid_search.cv_results_['mean_train_score'][i],
            'Mean Test Score': -grid_search.cv_results_['mean_test_score'][i],
            'Std Test Score': grid_search.cv_results_['std_test_score'][i],
            'Train Time': avg_train_time
        }
        
        # Save results to a CSV file in append mode
        results_df = pd.DataFrame([result])
        results_df.to_csv('xgboost_model_training_results.csv', mode='a', header=not os.path.isfile('xgboost_model_training_results.csv'), index=False)

    # Store best results
    best_model = grid_search.best_estimator_
    cv_score = -grid_search.best_score_

    # Predict on the test set and evaluate
    y_pred = best_model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"Training size: {size}, Best Params: {grid_search.best_params_}, CV MSE: {cv_score}, Test MSE: {test_mse}, Test R2: {test_r2}, Train Time: {avg_train_time}")

# Save the best model
best_model_idx = np.argmin([result['Mean Test Score'] for result in results])
best_model = grid_search.best_estimator_
dump(best_model, 'best_xgboost_model.joblib')

# Plot performance of each model tested
plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(result['Train Size'], result['Mean Test Score'], 'o-', label=f"Params: {result['Params']}")
plt.xlabel('Training Set Size')
plt.ylabel('Cross-Validation MSE')
plt.title('Model Performance During Grid Search and Cross-Validation')
plt.legend()
plt.grid(True)
plt.savefig('xgboost_model_performance.png')
plt.show()

# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.yticks(range(X.shape[1]), np.array(X.columns)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for XGBoost Regressor')
plt.savefig('feature_importance.png')
plt.show()
