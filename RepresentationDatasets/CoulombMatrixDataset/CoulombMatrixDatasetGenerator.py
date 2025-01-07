import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the dataset
file_path = 'GeneticAlgoRuns/CyclicMoleculeBenchmarkingDataset_NISTWEBBOOK.csv'
dataset = pd.read_csv(file_path)

# Column containing SMILES strings
smiles_column = 'SMILES'  # Update if the column name is different
if smiles_column not in dataset.columns:
    raise ValueError(f"Column '{smiles_column}' not found in dataset.")

def generate_coulomb_matrix_with_etkdg(smiles, max_atoms=50):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate conformer using the ETKDG method
        params = AllChem.ETKDGv3()  # Use the latest version of ETKDG
        params.randomSeed = 42  # Set a seed for reproducibility
        if not AllChem.EmbedMolecule(mol, params) == 0:
            return None  # Failed to generate conformer
        
        # Optimize the conformer
        AllChem.UFFOptimizeMolecule(mol)
        
        # Get atomic numbers and 3D coordinates
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(len(atomic_numbers))])
        
        # Compute the Coulomb matrix
        n_atoms = len(atomic_numbers)
        matrix = np.zeros((max_atoms, max_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
                else:
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist > 0:
                        matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / dist
        
        return matrix
    except Exception as e:
        return None

# Generate Coulomb matrices for all SMILES
coulomb_matrices = []
for smiles in dataset[smiles_column]:
    if pd.notnull(smiles):
        cm = generate_coulomb_matrix_with_etkdg(smiles)
        coulomb_matrices.append(cm)

# Save Coulomb matrices
output_path = 'CoulombMatrices.npy'
np.save(output_path, np.array(coulomb_matrices, dtype=object))

print(f"Coulomb matrices saved to: {output_path}")
