import pandas as pd
import numpy as np
import joblib
import selfies as sf
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "Dataset_with_SELFIES.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Define column names
selfies_col = "SELFIES"  # Ensure this column exists

properties = [
    'Thermal_Conductivity_40C',
    'Thermal_Conductivity_100C',
    'Density_40C',
    'Density_100C',
    'Heat_Capacity_40C',
    'Heat_Capacity_100C',
    'Viscosity_40C',
    'Viscosity_100C',
]

# Function to tokenize SELFIES strings into numerical format
def tokenize_selfies(selfies_list):
    """Convert SELFIES strings into integer tokenized sequences."""
    unique_tokens = set()  # Unique SELFIES tokens

    # Extract unique tokens from SELFIES
    for s in selfies_list:
        tokens = list(sf.split_selfies(s))  # Correct function usage
        unique_tokens.update(tokens)

    # Create mapping dictionaries
    token_to_int = {token: idx + 1 for idx, token in enumerate(sorted(unique_tokens))}  # +1 for padding

    # Convert SELFIES into integer sequences
    tokenized = [[token_to_int[token] for token in list(sf.split_selfies(s))] for s in selfies_list]
    
    return tokenized, len(unique_tokens) + 1  # Vocabulary size (+1 for padding)

# Determine max sequence length from the entire dataset
tokenized_selfies, vocab_size = tokenize_selfies(df[selfies_col].dropna().tolist())
max_seq_length = max(len(seq) for seq in tokenized_selfies)

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
def build_mlp(input_dim, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=16, input_length=input_dim))
    model.add(Flatten())

    # Add dense layers
    for _ in range(mlp_params["num_layers"]):
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
    
    # Drop NaN values only for this property
    prop_df = df[[property_name, selfies_col]].dropna()

    if prop_df.empty:
        print(f"Skipping {property_name} due to missing values.")
        return

    # Convert SELFIES to tokenized format
    tokenized, _ = tokenize_selfies(prop_df[selfies_col].tolist())
    X = pad_sequences(tokenized, maxlen=max_seq_length, padding="post")
    y = prop_df[property_name].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = []
    mae_scores = []
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_mlp(X_train.shape[1], vocab_size)
        model.fit(X_train, y_train, epochs=mlp_params["epochs"], batch_size=mlp_params["batch_size"], verbose=0)

        y_pred = model.predict(X_test).flatten()

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))

    # Save cross-validation results
    cv_results.append({
        "Property": property_name,
        "Mean R^2": np.mean(r2_scores),
        "Mean MAE": np.mean(mae_scores),
        "Mean MSE": np.mean(mse_scores)
    })

    # Train final model on full dataset
    final_model = build_mlp(X.shape[1], vocab_size)
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
        model = tf.keras.models.load_model(model_path)
    except:
        print(f"Skipping SHAP analysis for {property_name} (model not trained).")
        return

    # Drop NaN values only for this property
    prop_df = df[[property_name, selfies_col]].dropna(subset=[property_name])

    if prop_df.empty:
        print(f"Skipping SHAP analysis for {property_name} due to missing values.")
        return

    # Convert SELFIES to tokenized format
    tokenized, _ = tokenize_selfies(prop_df[selfies_col].tolist())
    X_sample = pad_sequences(tokenized, maxlen=max_seq_length, padding="post")

    # Compute SHAP values using KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # Convert SHAP values into a DataFrame
    shap_df = pd.DataFrame(shap_values)

    # Save SHAP values to CSV
    shap_filename = f"shap_values_{property_name}.csv"
    shap_df.to_csv(shap_filename, index=False)

    print(f"SHAP values for {property_name} saved to {shap_filename}")

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_sample)

# Run SHAP analysis for each property
for prop in properties:
    explain_model(prop)

print("Optimized SHAP analysis completed.")
