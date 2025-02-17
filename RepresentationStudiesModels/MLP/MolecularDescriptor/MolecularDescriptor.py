import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "Concatenated_Molecular_Descriptors.csv"  # Update with correct file path
df = pd.read_csv(file_path, low_memory=False)

# Define column names
smiles_col = "SMILES"
properties = [
    "Thermal_Conductivity_40C", "Thermal_Conductivity_100C",
    "Density_40C", "Density_100C",
    "Log_Viscosity_40C", "Log_Viscosity_100C",
    "Log_Heat_Capacity_40C", "Log_Heat_Capacity_100C"
]

# Identify descriptor columns (excluding SMILES and all target properties)
descriptor_cols = [col for col in df.columns if col not in properties and col != smiles_col]

# Prepare CSV file for cross-validation results
cv_results = []

# MLP Model Hyperparameters
mlp_params = {
    "optimizer": "adam",
    "dense_units": 16,
    "num_layers": 2,
    "epochs": 100,
    "batch_size": 64
}

# Function to build an MLP model
def build_mlp(input_dim):
    model = Sequential()
    model.add(Dense(mlp_params["dense_units"], activation="relu", input_shape=(input_dim,)))

    # Add hidden layers
    for _ in range(mlp_params["num_layers"] - 1):
        model.add(Dense(mlp_params["dense_units"], activation="relu"))

    # Output layer
    model.add(Dense(1, activation="linear"))

    # Compile model
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    return model

# Function to train MLP model for a given property
def train_mlp(property_name):
    if property_name not in df.columns:
        print(f"Skipping {property_name} (not found in dataset).")
        return
    
    # **Drop NaN values only for this property, keeping max data for others**
    prop_df = df[[property_name] + descriptor_cols].dropna()

    if prop_df.empty:
        print(f"Skipping {property_name} due to missing values.")
        return

    # Extract X (descriptors) and y (target property)
    X = prop_df[descriptor_cols].values
    y = prop_df[property_name].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = []
    mae_scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=mlp_params["epochs"], batch_size=mlp_params["batch_size"], verbose=0)

        y_pred = model.predict(X_test).flatten()

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

    # Save cross-validation results
    cv_results.append({
        "Property": property_name,
        "Mean R²": np.mean(r2_scores),
        "Mean MAE": np.mean(mae_scores),
        "Mean MSE": np.mean(mse_scores)
    })

    # Train final model on full dataset
    final_model = build_mlp(X.shape[1])
    final_model.fit(X, y, epochs=mlp_params["epochs"], batch_size=mlp_params["batch_size"], verbose=0)

    # Save trained model
    model_path = f"mlp_model_{property_name}.h5"
    final_model.save(model_path)

    print(f"Model for {property_name} trained and saved at {model_path}.")

# Train MLP models for each thermophysical property
for prop in properties:
    train_mlp(prop)

# Save cross-validation results to CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv("mlp_cross_validation_results.csv", index=False)

# Dictionary to store SHAP analysis results
shap_results = {}

# Function to perform SHAP analysis
def explain_model(property_name):
    try:
        model_path = f"mlp_model_{property_name}.h5"
        model = keras.models.load_model(model_path)
    except:
        print(f"Skipping SHAP analysis for {property_name} (model not trained).")
        return

    # **Drop NaN values only for this property, keeping max data for others**
    prop_df = df[[property_name] + descriptor_cols].dropna(subset=[property_name])

    if prop_df.empty:
        print(f"Skipping SHAP analysis for {property_name} due to missing values.")
        return

    # Randomly select 15% of the data for SHAP analysis
    prop_df_sample = prop_df.sample(frac=0.15, random_state=42)
    X_sample = prop_df_sample[descriptor_cols].values

    # Use k-means clustering to summarize the background dataset (faster SHAP computation)
    background = shap.kmeans(X_sample, 50)  

    # Compute SHAP values using KernelExplainer
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_sample)

    # Convert SHAP values into a DataFrame
    shap_df = pd.DataFrame(shap_values, columns=descriptor_cols)

    # Save SHAP values to CSV
    shap_filename = f"shap_values_{property_name}.csv"
    shap_df.to_csv(shap_filename, index=False)
    shap_results[property_name] = shap_filename

    print(f"SHAP values for {property_name} saved to {shap_filename}")

    # Plot top 20 influential descriptors
    mean_shap_values = np.abs(shap_df).mean().sort_values(ascending=False)
    top_20_shap = mean_shap_values[:20]

    plt.figure(figsize=(10, 6))
    top_20_shap.plot(kind="barh", color="skyblue")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Descriptors")
    plt.title(f"Top 20 Most Influential Descriptors for {property_name}")
    plt.gca().invert_yaxis()

    # Save the plot
    plot_filename = f"shap_plot_{property_name}.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()

    print(f"SHAP plot for {property_name} saved to {plot_filename}")

# Run SHAP analysis for each property
for prop in properties:
    explain_model(prop)

print("Optimized SHAP analysis completed.")
