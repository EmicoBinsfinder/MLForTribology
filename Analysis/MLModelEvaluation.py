import pandas as pd
import sys
sys.path.append('F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100')


# Define the file paths and their associated model names
file_paths = {
    "CNN": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/CNN_results_Thermal_Conductivity_100C.csv",
    "GMM": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/GMM_results_Thermal_Conductivity_100C.csv",
    "KNN": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/KNN_results_Thermal_Conductivity_100C.csv",
    "MLP": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/MLP_results_Thermal_Conductivity_100C.csv",
    "RF": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/RF_results_Thermal_Conductivity_100C.csv",
    "SVM": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/SVM_results_Thermal_Conductivity_100C.csv",
    "XGBoost": "F:/PhD/HIGH_THROUGHPUT_STUDIES/PropertyPrediction/TC100/XGBoost_results_Thermal_Conductivity_100C.csv"
}

# Function to find the best RMSE score and parameters from each file
def find_best_rmse(file_path, model_name):
    # Load the CSV file
    df = pd.read_csv(file_path)
    # Look for the column containing RMSE (or equivalent)
    rmse_column = [col for col in df.columns if "RMSE" in col or "Score" in col or "Error" in col or "MSE" in col or "score" in col or "mse" in col][0]
    # Find the row with the lowest RMSE value
    best_row = df.loc[df[rmse_column].idxmin()]  # Assuming lower RMSE is better
    return {
        "Model": model_name,
        "Best RMSE": best_row[rmse_column],
        "Parameters": {col: best_row[col] for col in df.columns if col != rmse_column}
    }

# Iterate over the files to process each model
results = []
for model_name, file_path in file_paths.items():
    try:
        result = find_best_rmse(file_path, model_name)
        results.append(result)
    except Exception as e:
        print(f"Error processing {model_name}: {e}")

# Convert the results into a DataFrame for easy visualization
summary_table = pd.DataFrame(results)

# Save the summary to a CSV file for later review
summary_table.to_csv("best_hyperparameters_summary.csv", index=False)

# Print the summary
print(summary_table)
