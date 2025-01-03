import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the dataset
file_path = 'GeneticAlgoRuns/CyclicMoleculeBenchmarkingDataset_NISTWEBBOOK.csv'
dataset = pd.read_csv(file_path)

# Ensure the column name for SMILES is correct (e.g., 'SMILES')
smiles_column = 'SMILES'  # Change this if the column name is different
if smiles_column not in dataset.columns:
    raise ValueError(f"Column '{smiles_column}' not found in dataset.")

# Function to compute Morgan fingerprints
def compute_morgan_fingerprint(smiles, radius=4, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fingerprint)
    except Exception as e:
        return None

# Compute fingerprints
dataset['MorganFingerprint'] = dataset[smiles_column].apply(
    lambda x: compute_morgan_fingerprint(x) if pd.notnull(x) else None
)

# Expand fingerprints into separate columns
fingerprint_df = pd.DataFrame(dataset['MorganFingerprint'].tolist(), columns=[f'FP_{i}' for i in range(2048)])
dataset = pd.concat([dataset, fingerprint_df], axis=1)

# Drop the original fingerprint column to keep only expanded features
dataset = dataset.drop(columns=['MorganFingerprint'])

# Save the dataset with fingerprints
output_path = 'Dataset_with_Morgan_Fingerprints.csv'
dataset.to_csv(output_path, index=False)

print(f"Dataset with Morgan fingerprints saved to: {output_path}")
