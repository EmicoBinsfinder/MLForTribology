import pandas as pd
from rdkit import Chem
import networkx as nx
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load dataset
# file_path = 'GeneticAlgoRuns/CyclicMoleculeBenchmarkingDataset_NISTWEBBOOK.csv'
# dataset = pd.read_csv(file_path)

# # Column containing SMILES strings
# smiles_column = 'SMILES'  # Update if the column name is different
# if smiles_column not in dataset.columns:
#     raise ValueError(f"Column '{smiles_column}' not found in dataset.")

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

# # Create molecular graphs and associate with properties
# graphs_with_properties = []
# for idx, row in dataset.iterrows():
#     if pd.notnull(row[smiles_column]):
#         graph = create_molecular_graph(row[smiles_column])
#         if graph is not None:
#             # Include the graph and all other properties in a dictionary
#             graph_data = {
#                 "graph": nx.node_link_data(graph),
#                 "properties": row.drop(smiles_column).to_dict()
#             }
#             graphs_with_properties.append(graph_data)

# # Save graphs and properties to JSON
# output_path = 'MolecularGraphs_with_Properties.json'

json_path = 'RepresentationDatasets/2DGraphDataset/MolecularGraphs_with_Properties.json'


with open(json_path, 'r') as f:
    molecular_graphs = json.load(f)

# Select an example graph (e.g., the first graph in the dataset)
example_graph_data = molecular_graphs[0]["graph"]  # Adjust index to select a different graph
example_properties = molecular_graphs[0]["properties"]

def plot_interactive_molecular_graph_with_links(graph_data, title="Interactive Molecular Graph"):
    # Convert node-link data back to a NetworkX graph
    graph = nx.node_link_graph(graph_data)
    
    # Extract positions for nodes (spring layout)
    pos = nx.spring_layout(graph)
    
    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    edge_text = []  # To hold bond info for tooltips
    edge_annotations = []  # To display bond type as text labels
    
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # Add bond properties to tooltip
        edge_text.append(f"Bond Properties:<br>{'<br>'.join([f'{k}: {v}' for k, v in data.items()])}")
        
        # Add annotations for bond type
        bond_label = data.get("bond_type", "")
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                text=bond_label,
                showarrow=False,
                font=dict(size=10, color="red")
            )
        )
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='text',
        mode='lines',
        text=edge_text
    )
    
    node_x = []
    node_y = []
    node_text = []  # To hold atom info for tooltips
    
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Add atom properties to tooltip
        node_text.append(f"Node Properties:<br>{'<br>'.join([f'{k}: {v}' for k, v in data.items()])}")
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=20,
            color='skyblue',
            line=dict(width=2, color='darkblue')
        )
    )
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            width=800,
            height=600,
            annotations=edge_annotations
        )
    )
    
    fig.show()

# Plot the selected graph
plot_interactive_molecular_graph_with_links(example_graph_data, title="Interactive Molecular Graph with Link Data")

