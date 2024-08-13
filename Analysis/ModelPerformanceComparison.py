import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load all the CSV files
files = {
    "RF": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/RF_cross_validation_results.csv",
    "SVM": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/SVM_cross_validation_results_svm.csv",
    "CNN": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/cnn_cv_results_with_best_dropout.csv",
    "GMM": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/cross_validation_results_gmm.csv",
    "XGBoost": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/cross_validation_results_xgboost.csv",
    "KNN": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/knn_performance_results.csv",
    "MLP": "C:/Users/eeo21/Desktop/MLAlgoTrainingImages/cv_results_with_dropout.csv"
}

dataframes = {name: pd.read_csv(path) for name, path in files.items()}

# Convert MSE to RMSE where necessary, and also convert standard deviation
for name, df in dataframes.items():
    if 'MSE' in df.columns:
        df['RMSE'] = np.sqrt(df['MSE'])
        if 'Standard Deviation' in df.columns:
            df['Std RMSE'] = df['Standard Deviation'] / (2 * np.sqrt(df['MSE']))
    if 'Average Test MSE' in df.columns:
        df['RMSE'] = np.sqrt(df['Average Test MSE'])
        if 'Std Test MSE' in df.columns:
            df['Std RMSE'] = df['Std Test MSE'] / (2 * np.sqrt(df['Average Test MSE']))

# Normalize training set size to percentage
for name, df in dataframes.items():
    size_col = df.columns[0]
    if df[size_col].max() <= 1:  # Assuming the training set size is between 0 and 1
        df[size_col] = df[size_col] * 100

# Prepare data for plotting
plot_data = {}
for name, df in dataframes.items():
    if 'RMSE' in df.columns:
        if 'Std RMSE' in df.columns:
            std_col = 'Std RMSE'
        else:
            std_col = None
        
        plot_data[name] = {
            'size': df.iloc[:, 0],
            'rmse': df['RMSE'],
            'std': df[std_col] if std_col else np.zeros(len(df))
        }

# Plot the data
plt.figure(figsize=(12, 8))

for name, data in plot_data.items():
    best_index = data['rmse'].idxmin()
    best_rmse = data['rmse'][best_index]
    best_size = data['size'][best_index]
    best_std = data['std'][best_index]
    label = f"{name} (Best RMSE: {best_rmse:.2f} ± {best_std:.2f} at {best_size:.1f}%)"
    plt.errorbar(data['size'], data['rmse'], yerr=data['std'], label=label, capsize=5, marker='o')

plt.xlabel('Training Set Size (%)', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.title('Performance vs. Training Set Size', fontsize=16)
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0.5, 100.5)
plt.ylim(0, 15)
plt.grid(color='grey', linestyle='--', linewidth=0.5)
plt.grid(which="minor", linestyle='--', linewidth=0.2)
plt.minorticks_on()
plt.legend(loc='best')
plt.grid(True)
plt.savefig('C:/Users/eeo21/Desktop/MLAlgoTrainingImages/ModelPerformanceComparison.png')
plt.show()
