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

### Load in original dataset
OriginalDataset = pd.read_csv('Dataset/OriginalDataset.csv', index_col=False)

### Remove molucules with Viscosities that are too low or too high

ViscMask = (OriginalDataset['visco@40C[cP]'] >= 2) & (OriginalDataset['visco@40C[cP]'] <= 50)

Visc40FilteredDataset = OriginalDataset[ViscMask]

### Remove entries with more than 2 rings

def GetNumRings(SMILES):
    Mol = Chem.MolFromSmiles(SMILES)
    NumRings = Mol.GetRingInfo().NumRings()
    return NumRings

Visc40FilteredDataset['NumRings'] = Visc40FilteredDataset['smiles'].apply(GetNumRings)
RingMask = (Visc40FilteredDataset['NumRings'] <=2)
RingsFilteredDataset = Visc40FilteredDataset[RingMask]

### Create ECFP Fingerprints for every entry
# Will use radius of 8 as this is shown to provide fully unique representations (can validate this ourselves)

def GetFingerprint(SMILES):
    Mol = Chem.MolFromSmiles(SMILES)
    radius = 2
    nBits = 2048  # Length of the bit vector
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(Mol, radius, nBits)
    # ecfp4_list = list(ecfp4)
    return ecfp4

# RingsFilteredDataset['SELFIES'] = RingsFilteredDataset['smiles'].apply(GetFingerprint)

### Create SELFIES representation for every entry
def GetSELFIES(SMILES):
    return sf.encoder(SMILES)

RingsFilteredDataset['SELFIES'] = RingsFilteredDataset['smiles'].apply(GetSELFIES)

### Remove entries where Viscosity at 100C is larger than viscosity at 40C
condition = RingsFilteredDataset['visco@100C[cP]'] <= RingsFilteredDataset['visco@40C[cP]']
RingsFilteredDataset['WhackVisc'] = RingsFilteredDataset['visco@100C[cP]'].where(condition)
FilteredDataset = RingsFilteredDataset.dropna(subset=['WhackVisc'])

FilteredDataset.to_csv('Dataset/FinalDataset.csv', index=False)

# Function to convert SMILES to descriptors
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    calculator = MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calculator.CalcDescriptors(mol)
    return descriptors
