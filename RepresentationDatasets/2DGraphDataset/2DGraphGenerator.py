import pandas as pd
from rdkit import Chem
import networkx as nx
import json

# Load dataset
file_path = 'GeneticAlgoRuns/CyclicMoleculeBenchmarkingDataset_NISTWEBBOOK.csv'
dataset = pd.read_csv(file_path)

# Column containing SMILES strings
smiles_column = 'SMILES'  # Update if the column name is different
if smiles_column not in dataset.columns:
    raise ValueError(f"Column '{smiles_column}' not found in dataset.")

# Function to create molecular graphs
def create_molecular_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create a NetworkX graph
        graph = nx.Graph()
        
        # Add nodes (atoms)
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), 
                           atomic_num=atom.GetAtomicNum(),
                           aromatic=atom.GetIsAromatic(),
                           hybridization=str(atom.GetHybridization()),
                           degree=atom.GetDegree(),
                           formal_charge=atom.GetFormalCharge())
        
        # Add edges (bonds)
        for bond in mol.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(),
                           bond.GetEndAtomIdx(),
                           bond_type=str(bond.GetBondType()),
                           is_aromatic=bond.GetIsAromatic())
        
        return graph
    except Exception as e:
        return None

# Create molecular graphs and associate with properties
graphs_with_properties = []
for idx, row in dataset.iterrows():
    if pd.notnull(row[smiles_column]):
        graph = create_molecular_graph(row[smiles_column])
        if graph is not None:
            # Include the graph and all other properties in a dictionary
            graph_data = {
                "graph": nx.node_link_data(graph),
                "properties": row.drop(smiles_column).to_dict()
            }
            graphs_with_properties.append(graph_data)

# Save graphs and properties to JSON
output_path = 'MolecularGraphs_with_Properties.json'
with open(output_path, 'w') as f:
    json.dump(graphs_with_properties, f, indent=4)

print(f"Molecular graphs with properties saved to: {output_path}")
