"""
Author: Egheosa Ogbomo

Script to analyse types of molecules in Kajita dataset

- Count of different molecules
- Spread of viscosities
- Spread of C-Number
- How ML predictive power varies depending on different molecule class

Groups to check.

- Cyclic Aliphatic
- Ether
- Ester
- Aromatics
- Paraffins
- Aromatic Esters
- Alkenes
- Ketones
- Napthalenes

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings, CalcNumBridgeheadAtoms, CalcNumSaturatedCarbocycles

def contains_oxygen(smiles_string):
    # Convert the SMILES string to a molecule
    molecule = Chem.MolFromSmiles(smiles_string)
    
    # Check if the molecule contains oxygen (atomic number 8)
    if molecule:
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 8:  # Atomic number 8 is oxygen
                return True
    return False

def contains_ester(smiles_string):
    # Convert the SMILES string to a molecule
    molecule = Chem.MolFromSmiles(smiles_string)
    
    # Define the ester group as a substructure in SMILES
    ester_substructure = Chem.MolFromSmarts('C(=O)O')
    
    # Check if the molecule contains the ester substructure
    if molecule.HasSubstructMatch(ester_substructure):
        return True
    return False

Dataset = pd.read_csv('C:/Users/eeo21/VSCodeProjects/MLForTribology/Analysis/KajitaDatasetProperties.csv')
# SmilesList = Dataset['smiles']

# PropertiesList = []
# NumHeavyAtomsList = []

# for SMILES in SmilesList:
    
#     # Convert to RDKit Mol Object
#     Mol = Chem.MolFromSmiles(SMILES)
#     Chem.SanitizeMol(Mol)

#     NumHeavyAtoms = Mol.GetNumHeavyAtoms()
#     NumHeavyAtomsList.append(NumHeavyAtoms)

#     # Check for Aromatic
#     if CalcNumAromaticRings(Mol) > 0:
#         # Check if Aromatic Napthalene
#         if CalcNumBridgeheadAtoms(Mol) > 0:
#             PropertiesList.append('Aromatic Napthalene')
#             continue
        
#         # Check if Aromatic Ester
#         elif contains_ester(SMILES):
#             PropertiesList.append('Aromatic Ester')
#             continue

#         # Check if Aromatic Ether
#         elif contains_oxygen(SMILES):
#             PropertiesList.append('Aromatic Ether')
#             continue

#         else:
#             PropertiesList.append('Aromatic')
#             continue

#     # Check if a Saturated Ring
#     if CalcNumSaturatedCarbocycles(Mol) > 0:
#         if contains_ester(SMILES):
#             PropertiesList.append('Cyclic Ester')
#             continue
#         if contains_oxygen(SMILES):
#             PropertiesList.append('Cyclic Ether')
#             continue
#         else:
#             PropertiesList.append('Cyclic Paraffin')
#             continue

#     # Check if Napthalene
#     if CalcNumBridgeheadAtoms(Mol) > 0:
#         PropertiesList.append('Aromatic Napthalene')
#         continue

#     # Check if Ester
#     if contains_ester(SMILES):
#         PropertiesList.append('Ester')
#         continue

#     # Check if Ether
#     if contains_oxygen(SMILES):
#         PropertiesList.append('Ether')
#         continue

#     else:
#         PropertiesList.append('Paraffin')

# Dataset['Class'] = PropertiesList
# Dataset['HeavyAtoms'] = NumHeavyAtomsList

# Task 1: Create a table showing how many of each different Class of molecule there is
class_counts = Dataset['Class'].value_counts()

# Display the table of Class counts
print(class_counts)

# Task 2: Plotting Number of Heavy Atoms against Viscosity at 40°C and 100°C, color-coded by class
# Define colors for each class
classes = Dataset['Class'].unique()
colors = plt.cm.get_cmap('tab10', len(classes))

plt.figure(figsize=(14, 6))

# Plot for 40°C
plt.subplot(1, 2, 1)
for i, class_type in enumerate(classes):
    subset = Dataset[Dataset['Class'] == class_type]
    plt.scatter(subset['HeavyAtoms'], subset['visco@40C[cP]'], label=class_type, color=colors(i))
plt.title('Heavy Atoms vs Viscosity @ 40°C (Color-coded by Class)')
plt.xlabel('Number of Heavy Atoms')
plt.ylabel('Viscosity @ 40°C [cP]')
plt.legend()


# Plot for 100°C
plt.subplot(1, 2, 2)
for i, class_type in enumerate(classes):
    subset = Dataset[Dataset['Class'] == class_type]
    plt.scatter(subset['HeavyAtoms'], subset['visco@100C[cP]'], label=class_type, color=colors(i))
plt.title('Heavy Atoms vs Viscosity @ 100°C (Color-coded by Class)')
plt.xlabel('Number of Heavy Atoms')
plt.ylabel('Viscosity @ 100°C [cP]')
plt.legend()

plt.tight_layout()
plt.savefig('ViscSpreadKajita.png')
plt.show()

# Task 3: Bar chart showing the spread of different molecules with different numbers of heavy atoms
heavy_atoms_counts = Dataset['HeavyAtoms'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
heavy_atoms_counts.plot(kind='bar', color='blue')
plt.title('Distribution of Molecules by Number of Heavy Atoms')
plt.xlabel('Number of Heavy Atoms')
plt.ylabel('Count of Molecules')
plt.grid(True)
plt.savefig('HeavyAtomSpread.png')
plt.show()



