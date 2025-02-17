import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from mordred import Calculator, descriptors

# Load dataset
df = pd.read_csv("C:/Users/eeo21/Desktop/Datasets/NISTCyclicMoleculeDataset_LogScale.csv")  # Change filename if needed

# Extract SMILES strings and target properties
smiles_list = df["SMILES"].tolist()
target_properties = df.drop(columns=["SMILES", "Molecule Name", "C-Number"], errors='ignore')

# Initialize Mordred descriptor calculators
calc_2D = Calculator(descriptors, ignore_3D=True)
calc_3D = Calculator(descriptors, ignore_3D=False)

# Function to compute RDKit 2D descriptors
def compute_rdkit_2d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {desc_name: func(mol) for desc_name, func in Descriptors.descList}
    return {}

# Function to compute 3D descriptors (requires conformers)
def compute_rdkit_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            return {
                "MolecularVolume": rdMolDescriptors.CalcExactMolWt(mol),
                "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration(mol),
                "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol),
            }
        except:
            return {}  # Return empty dictionary if conformer generation fails
    return {}

# Compute 2D and 3D descriptors
# rdkit_2d_descs = [compute_rdkit_2d(smiles) for smiles in smiles_list]
mordred_2d_descs = [calc_2D(Chem.MolFromSmiles(smiles)).asdict() if Chem.MolFromSmiles(smiles) else {} for smiles in smiles_list]

# rdkit_3d_descs = [compute_rdkit_3d(smiles) for smiles in smiles_list]
mordred_3d_descs = [calc_3D(Chem.MolFromSmiles(smiles)).asdict() if Chem.MolFromSmiles(smiles) else {} for smiles in smiles_list]

# Convert to DataFrames
# rdkit_2d_df = pd.DataFrame(rdkit_2d_descs)
mordred_2d_df = pd.DataFrame(mordred_2d_descs)
# rdkit_3d_df = pd.DataFrame(rdkit_3d_descs)
mordred_3d_df = pd.DataFrame(mordred_3d_descs)

# Merge 2D descriptors and target properties
desc_2d_df = pd.concat([df["SMILES"], mordred_2d_df, target_properties], axis=1)
desc_3d_df = pd.concat([df["SMILES"], mordred_3d_df, target_properties], axis=1)

# Function to clean descriptor dataset (retain SMILES)
def clean_descriptor_dataset(df):
    # Keep SMILES column
    smiles_column = df["SMILES"]

    # Keep only float columns
    float_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_cleaned = df[float_columns]

    # Drop columns where all values are zero
    df_cleaned = df_cleaned.loc[:, (df_cleaned != 0).any(axis=0)]

    # Reattach SMILES column
    df_cleaned.insert(0, "SMILES", smiles_column)

    return df_cleaned

# Clean both datasets
# desc_2d_cleaned = clean_descriptor_dataset(desc_2d_df)
# desc_3d_cleaned = clean_descriptor_dataset(desc_3d_df)

# Save cleaned datasets
desc_2d_df.to_csv("2D_Molecular_Descriptors_Cleaned.csv", index=False)
desc_3d_df.to_csv("3D_Molecular_Descriptors_Cleaned.csv", index=False)

print("✅ 2D and 3D molecular descriptors generated and cleaned successfully!")
print("📂 Files saved as: 2D_Molecular_Descriptors_Cleaned.csv, 3D_Molecular_Descriptors_Cleaned.csv")
