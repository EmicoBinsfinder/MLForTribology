"""
Script to prepare dataset for use with various ML algorithms:


- Provides stronger weighting to molecules with properties from experiments

"""

### Imports
import pandas as pd
import csv
import sklearn
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Load the datasets
test_df = pd.read_csv('Datasets/EBDatasetSMILES.csv')
train_df = pd.read_csv('Datasets/KajitaDatasetProperties.csv')

# Get all available descriptor names from the rdkit.Chem.Descriptors module
all_descriptors = [desc_name[0] for desc_name in Descriptors.descList]

# Create a MolecularDescriptorCalculator object for all descriptors
calc = MolecularDescriptorCalculator(all_descriptors)

# Function to calculate all molecular descriptors from SMILES strings
def calculate_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return pd.Series(calc.CalcDescriptors(mol), index=all_descriptors)
    else:
        return pd.Series([None] * len(all_descriptors), index=all_descriptors)

# Apply descriptor calculation to the SMILES (test) dataset
test_df[all_descriptors] = test_df['SMILES'].apply(calculate_all_descriptors)


# # Create the new SMILES (test) dataset with descriptors and SMILES
# test_dataset = test_df[['SMILES'] + test_df[all_descriptors] + test_df['Experimental_40C_Viscosity'], test_df['Experimental_100C_Viscosity']]

# Apply descriptor calculation to the Kajita (train) dataset
train_df[all_descriptors] = train_df['smiles'].apply(calculate_all_descriptors)

# Create the new Kajita (train) dataset with descriptors, SMILES, and viscosities at 40C and 100C
# train_dataset = train_df[['smiles'] + all_descriptors + ['Viscosity_40C', 'Viscosity_100C']]

# Save both datasets to CSV files
train_df.to_csv('Datasets/Experimental_Test_Dataset_Descriptors.csv', index=False)
test_df.to_csv('Datasets/Kajita_Train_Dataset_Descriptors.csv', index=False)


### Load in original dataset
# OriginalDataset = pd.read_csv('Datasets/OriginalDataset.csv', index_col=False)

# ### Remove molucules with Viscosities that are too low or too high

# ViscMask = (OriginalDataset['visco@40C[cP]'] >= 2) & (OriginalDataset['visco@40C[cP]'] <= 50)

# Visc40FilteredDataset = OriginalDataset[ViscMask]

# ### Remove entries with more than 2 rings

# def GetNumRings(SMILES):
#     Mol = Chem.MolFromSmiles(SMILES)
#     NumRings = Mol.GetRingInfo().NumRings()
#     return NumRings

# Visc40FilteredDataset['NumRings'] = Visc40FilteredDataset['smiles'].apply(GetNumRings)
# RingMask = (Visc40FilteredDataset['NumRings'] <=2)
# RingsFilteredDataset = Visc40FilteredDataset[RingMask]

# ### Create ECFP Fingerprints for every entry
# # Will use radius of 8 as this is shown to provide fully unique representations (can validate this ourselves)

# def GetFingerprint(SMILES):
#     Mol = Chem.MolFromSmiles(SMILES)
#     radius = 2
#     nBits = 2048  # Length of the bit vector
#     ecfp4 = AllChem.GetMorganFingerprintAsBitVect(Mol, radius, nBits)
#     # ecfp4_list = list(ecfp4)
#     return ecfp4

# # RingsFilteredDataset['SELFIES'] = RingsFilteredDataset['smiles'].apply(GetFingerprint)

# ### Create SELFIES representation for every entry
# def GetSELFIES(SMILES):
#     return sf.encoder(SMILES)

# RingsFilteredDataset['SELFIES'] = RingsFilteredDataset['smiles'].apply(GetSELFIES)

# ### Remove entries where Viscosity at 100C is larger than viscosity at 40C
# condition = RingsFilteredDataset['visco@100C[cP]'] <= RingsFilteredDataset['visco@40C[cP]']
# RingsFilteredDataset['WhackVisc'] = RingsFilteredDataset['visco@100C[cP]'].where(condition)
# FilteredDataset = RingsFilteredDataset.dropna(subset=['WhackVisc'])

# FilteredDataset.to_csv('Datasets/FinalDataset.csv', index=False)

# ShuffledDataset = FilteredDataset.sample(frac=1, random_state=1).reset_index(drop=True)

# total_rows = len(ShuffledDataset)
# ten_percent_rows = int(total_rows * 0.1)
# ninety_percent_rows = int(total_rows * 0.9)

# ten_percent_df = ShuffledDataset.head(ten_percent_rows)
# last_ninety_percent_df = ShuffledDataset.tail(ninety_percent_rows)

# TwentyPercentDataset = last_ninety_percent_df.sample(frac=0.2, random_state=1).reset_index(drop=True)

# ten_percent_df.to_csv('Datasets/FinalTestDataset.csv')
# last_ninety_percent_df.to_csv('Datasets/LargeTrainingDataset.csv')
# TwentyPercentDataset.to_csv('Datasets/GridSearchDataset.csv')

# # Function to convert SMILES to descriptors
# def smiles_to_descriptors(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     calculator = MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
#     descriptors = calculator.CalcDescriptors(mol)
#     return descriptors

# ShuffledDataset = descriptors_df.sample(frac=1, random_state=1).reset_index(drop=True)

# total_rows = len(ShuffledDataset)
# ten_percent_rows = int(total_rows * 0.1)
# ninety_percent_rows = int(total_rows * 0.9)

# ten_percent_df = ShuffledDataset.head(ten_percent_rows)
# last_ninety_percent_df = ShuffledDataset.tail(ninety_percent_rows)

# TwentyPercentDataset = last_ninety_percent_df.sample(frac=0.2, random_state=1).reset_index(drop=True)

# ten_percent_df.to_csv('Datasets/FinalTestDataset_Descriptors.csv')
# last_ninety_percent_df.to_csv('Datasets/LargeTrainingDataset_Descriptors.csv')
# TwentyPercentDataset.to_csv('Datasets/GridSearchDataset_Descriptors.csv')