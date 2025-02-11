import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Load dataset
file_path = "TransformedDataset.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Define column names
smiles_col = "SMILES"
properties = [
    "Thermal_Conductivity_40C", "Thermal_Conductivity_100C",
    "Density_40C", "Density_100C",
    "Viscosity_40C", "Viscosity_100C",
    "Heat_Capacity_40C", "Heat_Capacity_100C"
]

# Create the fingerprint generator (Morgan radius=8, 2048-bit)
fpgen = GetMorganGenerator(radius=8, fpSize=2048)

def smiles_to_fingerprint(smiles):
    """Convert a SMILES string to a 2048-bit Morgan fingerprint with radius=8."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = fpgen.GetFingerprint(mol)  # Generate fingerprint
        return list(fp)  # Convert to list of bits
    else:
        return None

# Generate fingerprints for all molecules
df["Fingerprint"] = df[smiles_col].apply(smiles_to_fingerprint)

# Remove molecules where fingerprint generation failed
df = df.dropna(subset=["Fingerprint"])

# Convert fingerprint list to separate columns
fingerprint_df = pd.DataFrame(df["Fingerprint"].to_list(), index=df.index)

# Ensure all fingerprint columns are numeric
fingerprint_df = fingerprint_df.apply(pd.to_numeric, errors="coerce")

# Merge fingerprints with the dataset
df = pd.concat([df, fingerprint_df], axis=1)
df.drop(columns=["Fingerprint"], inplace=True)

# Remove any remaining NaNs in the dataset
df.dropna(inplace=True)

# Save cleaned fingerprints to CSV
fingerprint_df.to_csv("molecular_fingerprints.csv", index=False)

# Prepare CSV file for cross-validation results
cv_results = []

# Function to train and evaluate Random Forest with k-Fold Cross Validation
def train_rf(property_name):
    # Ensure the property exists in the dataset
    if property_name not in df.columns:
        print(f"Skipping {property_name} (not found in dataset).")
        return

    # Identify fingerprint columns (exclude known property columns)
    fingerprint_cols = [col for col in df.columns if col not in properties and col != "SMILES"]

    # Extract only the target column and fingerprint features
    prop_df = df[[property_name] + fingerprint_cols].copy()

    # Remove rows where the target column is NaN
    prop_df = prop_df.dropna(subset=[property_name])

    # Check if the dataset is empty after removing NaNs
    if prop_df.empty:
        print(f"Skipping {property_name} due to missing values.")
        return

    # Extract X (fingerprints) and y (target property)
    X = prop_df[fingerprint_cols].values
    y = prop_df[property_name].values

    # Ensure there are no NaNs before training
    if np.isnan(X).any() or np.isnan(y).any():
        print(f"Skipping {property_name} due to remaining NaNs.")
        return

    # k-Fold Cross Validation (5 folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Define Random Forest model with specified hyperparameters
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))  # Compute RMSE

    # Save cross-validation results
    cv_results.append({
        "Property": property_name,
        "Mean R^2": np.mean(r2_scores),
        "Mean MAE": np.mean(mae_scores),
        "Mean RMSE": np.mean(rmse_scores)
    })

    # Train final model on full dataset
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, f"rf_model_{property_name}.pkl")

    print(f"Random Forest Model for {property_name} trained and saved.")

# Train and evaluate Random Forest models for each property
for prop in properties:
    train_rf(prop)

# Save cross-validation results to CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv("rf_cross_validation_results.csv", index=False)

print("Cross-validation results saved to 'rf_cross_validation_results.csv'.")

# Perform SHAP analysis on 15% of the data
def explain_model(property_name):
    # Load trained model
    try:
        model = joblib.load(f"rf_model_{property_name}.pkl")
    except FileNotFoundError:
        print(f"Skipping SHAP analysis for {property_name} (model not trained).")
        return

    # Identify fingerprint columns
    fingerprint_cols = [col for col in df.columns if col not in properties and col != "SMILES"]

    # Extract the property data and fingerprints
    prop_df = df[[property_name] + fingerprint_cols].dropna(subset=[property_name])

    if prop_df.empty:
        print(f"Skipping SHAP analysis for {property_name} due to missing values.")
        return

    # Randomly select 15% of the data for SHAP analysis
    prop_df_sample = prop_df.sample(frac=0.15, random_state=42)
    X_sample = prop_df_sample[fingerprint_cols].values

    # Use k-means clustering to summarize the background dataset (faster SHAP)
    background = shap.kmeans(X_sample, 50)  # Summarize with 50 representative points

    # Compute SHAP values using optimized background
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # Convert SHAP values into a DataFrame
    shap_df = pd.DataFrame(shap_values, columns=fingerprint_cols)

    # Save SHAP values to CSV
    shap_filename = f"shap_values_{property_name}.csv"
    shap_df.to_csv(shap_filename, index=False)

    print(f"SHAP values for {property_name} saved to {shap_filename}")

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_sample, feature_names=fingerprint_cols)

# Run SHAP analysis for each property on 15% of the data
for prop in properties:
    explain_model(prop)

print("Optimized SHAP analysis completed on 15% of the dataset.")
