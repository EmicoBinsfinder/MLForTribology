import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import shap

# Load dataset
file_path = "NISTCyclicMoleculeDataset_LogScale.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Define column names
smiles_col = "SMILES"
properties = [
    "Thermal Conductivity_40C", "Thermal Conductivity_100C",
    "Density_40C", "Density_100C",
    "Log_Viscosity_40C", "Log_Viscosity_100C",
    "Log_Heat_Capacity_40C", "Log_Heat Capacity_100C"
]

# Function to generate Morgan fingerprints from SMILES
def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp)
    else:
        return None

# Generate fingerprints for all molecules
df["Fingerprint"] = df[smiles_col].apply(smiles_to_fingerprint)

# Remove molecules where fingerprint generation failed
df = df.dropna(subset=["Fingerprint"])

# Convert fingerprint list to separate columns
fingerprint_df = pd.DataFrame(df["Fingerprint"].to_list(), index=df.index)
df = pd.concat([df, fingerprint_df], axis=1)
df.drop(columns=["Fingerprint"], inplace=True)

# Save fingerprints to CSV
fingerprint_df.to_csv("molecular_fingerprints.csv", index=False)

# Function to train kNN model with predefined hyperparameters
def train_knn(property_name):
    # Drop missing values for the specific property
    prop_df = df.dropna(subset=[property_name])
    X = prop_df.iloc[:, len(properties):].values  # Fingerprint columns
    y = prop_df[property_name].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define kNN model with specified hyperparameters
    knn = KNeighborsRegressor(
        n_neighbors=21, algorithm="ball_tree", leaf_size=200, p=1, weights="distance"
    )
    knn.fit(X_train, y_train)

    # Save model
    joblib.dump(knn, f"knn_model_{property_name}.pkl")

    # Save hyperparameters
    hyperparams = {
        "algorithm": "ball_tree",
        "leaf_size": 200,
        "n_neighbors": 21,
        "p": 1,
        "weights": "distance"
    }
    with open(f"knn_hyperparams_{property_name}.txt", "w") as f:
        f.write(str(hyperparams))

    print(f"Model for {property_name} trained and saved.")

# Train kNN models for each thermophysical property
for prop in properties:
    train_knn(prop)

# Salience Study using SHAP
def explain_model(property_name):
    # Load trained model
    model = joblib.load(f"knn_model_{property_name}.pkl")
    
    # Extract data
    prop_df = df.dropna(subset=[property_name])
    X = prop_df.iloc[:, len(properties):].values
    
    # Compute SHAP values
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X[:100])  # Limit to 100 samples for efficiency

    # Plot SHAP summary
    shap.summary_plot(shap_values, X, feature_names=[f"Bit_{i}" for i in range(X.shape[1])])

# Run salience study for each property
for prop in properties:
    explain_model(prop)

print("Salience study completed.")
