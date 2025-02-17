import os
import pandas as pd
import numpy as np
import joblib
import selfies as sf
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load dataset
file_path = "Dataset_with_SELFIES.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Define column names
selfies_col = "SELFIES"
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

# Transformer Model Hyperparameters
transformer_params = {
    "head_size": 16,
    "num_heads": 50,
    "ff_dim": 64,
    "dropout": 0.0,
    "num_transformer_blocks": 2,
    "dense_units": 16,
    "optimizer": "adam",
    "epochs": 100,
    "batch_size": 32
}

# Function to tokenize SELFIES strings into numerical format
def tokenize_selfies(selfies_list):
    """Convert SELFIES strings into integer tokenized sequences."""
    unique_tokens = set()  # Unique SELFIES tokens

    # Extract unique tokens from SELFIES
    for s in selfies_list:
        try:
            tokens = list(sf.split_selfies(s))  # Standard SELFIES splitting
        except AttributeError:
            tokens = list(sf.decoder(s))  # Fallback to SMILES character tokens

        unique_tokens.update(tokens)

    # Create mapping dictionaries
    token_to_int = {token: idx + 1 for idx, token in enumerate(sorted(unique_tokens))}  # +1 for padding

    # Convert SELFIES into integer sequences
    tokenized = []
    for s in selfies_list:
        try:
            tokens = list(sf.split_selfies(s))
        except AttributeError:
            tokens = list(sf.decoder(s))  # Fallback

        tokenized.append([token_to_int[token] for token in tokens])
    
    return tokenized, len(unique_tokens) + 1  # Vocabulary size (+1 for padding)

# Determine max sequence length from the entire dataset
tokenized_selfies, vocab_size = tokenize_selfies(df[selfies_col].dropna().tolist())
max_seq_length = max(len(seq) for seq in tokenized_selfies)

# Prepare CSV file for cross-validation results
cv_results = []

# Function to build a Transformer model
def build_transformer(input_dim, vocab_size):
    inputs = Input(shape=(input_dim,))

    # Embedding layer
    x = Embedding(input_dim=vocab_size, output_dim=64, input_length=input_dim)(inputs)

    # Transformer Blocks
    for _ in range(transformer_params["num_transformer_blocks"]):
        attn_output = MultiHeadAttention(
            key_dim=transformer_params["head_size"],
            num_heads=transformer_params["num_heads"],
            dropout=transformer_params["dropout"]
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        ffn_output = Dense(transformer_params["ff_dim"], activation="relu")(x)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = GlobalAveragePooling1D()(x)
    x = Dense(transformer_params["dense_units"], activation="relu")(x)
    outputs = Dense(1, activation="linear")(x)

    # Build and compile model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    
    return model

# Function to train Transformer model for a given property
def train_transformer(property_name):
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

        model = build_transformer(X_train.shape[1], vocab_size)
        model.fit(X_train, y_train, epochs=transformer_params["epochs"], batch_size=transformer_params["batch_size"], verbose=0)

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
    final_model = build_transformer(X.shape[1], vocab_size)
    final_model.fit(X, y, epochs=transformer_params["epochs"], batch_size=transformer_params["batch_size"], verbose=0)

    # Save trained model
    model_path = f"transformer_model_{property_name}.h5"
    final_model.save(model_path)

    print(f"Model for {property_name} trained and saved at {model_path}.")

# Train Transformer models for each thermophysical property
for prop in properties:
    train_transformer(prop)

# Save cross-validation results to CSV
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv("transformer_cross_validation_results.csv", index=False)

print("Transformer training completed.")

def explain_model(property_name):
    try:
        # Load trained Transformer model
        model_path = f"transformer_model_{property_name}.h5"
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

    # Calculate mean absolute SHAP values per feature
    mean_shap_values = np.abs(shap_df).mean().sort_values(ascending=False)

    # Select top 20 influential SHAP values
    top_20_shap = mean_shap_values[:20]

    # Plot top 20 SHAP values
    plt.figure(figsize=(10, 6))
    top_20_shap.plot(kind="barh", color="skyblue")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("SELFIES Token Features")
    plt.title(f"Top 20 Most Influential SHAP Values for {property_name}")
    plt.gca().invert_yaxis()

    # Save the plot
    plot_filename = f"shap_plot_{property_name}.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()

    print(f"SHAP plot for {property_name} saved to {plot_filename}")

for prop in properties:
    explain_model(prop)