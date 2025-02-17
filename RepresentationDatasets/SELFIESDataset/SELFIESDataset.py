import pandas as pd
import selfies as sf

# Load the uploaded dataset
input_file = 'TransformedDataset.csv'
dataset = pd.read_csv(input_file)

# Specify the column containing SMILES strings
smiles_column = 'SMILES'  # Adjust if necessary

if smiles_column not in dataset.columns:
    raise ValueError(f"Column '{smiles_column}' not found in the dataset.")

# Convert SMILES to SELFIES
def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception as e:
        return None  # Handle invalid SMILES

dataset['SELFIES'] = dataset[smiles_column].apply(smiles_to_selfies)

# Save the dataset with SELFIES
output_file = 'Dataset_with_SELFIES.csv'
dataset.to_csv(output_file, index=False)

print(f"Dataset with SELFIES saved to: {output_file}")
