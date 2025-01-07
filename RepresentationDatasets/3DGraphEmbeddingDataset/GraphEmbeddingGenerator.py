import json
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd

# Load the JSON file with 3D molecular graphs
json_path = '3D_Molecular_Graphs.json'

with open(json_path, 'r') as f:
    molecular_graphs = json.load(f)

# Convert JSON graphs to NetworkX graphs
nx_graphs = []
properties = []

for graph_entry in molecular_graphs:
    nx_graph = nx.node_link_graph(graph_entry['graph'])  # Convert to NetworkX graph
    nx_graphs.append(nx_graph)
    properties.append(graph_entry['properties'])  # Save properties for downstream use

# Step 1: Weisfeiler-Lehman relabeling
def wl_relabel(graph, iterations=2):
    labels = {node: str(data['atomic_num']) for node, data in graph.nodes(data=True)}  # Initial labels
    
    for _ in range(iterations):
        new_labels = {}
        for node in graph.nodes:
            neighborhood = [labels[neighbor] for neighbor in graph.neighbors(node)]
            neighborhood.sort()
            new_labels[node] = labels[node] + "_" + "_".join(neighborhood)
        labels = new_labels
    return labels

# Step 2: Create subgraph "documents"
documents = []
for graph in nx_graphs:
    labels = wl_relabel(graph)
    document = [label for label in labels.values()]  # Collect all node labels as a document
    documents.append(document)

# Step 3: Train Word2Vec on the node label "documents"
model = Word2Vec(sentences=documents, vector_size=128, window=5, min_count=1, workers=4)

# Step 4: Generate graph embeddings by aggregating node embeddings
graph_embeddings = []
for document in documents:
    node_embeddings = [model.wv[node] for node in document if node in model.wv]
    if node_embeddings:
        graph_embedding = sum(node_embeddings) / len(node_embeddings)  # Average node embeddings
    else:
        graph_embedding = [0] * 128  # Default zero vector if no nodes
    graph_embeddings.append(graph_embedding)

# Convert embeddings to a DataFrame
embeddings_df = pd.DataFrame(graph_embeddings, columns=[f"dim_{i}" for i in range(128)])

# Add properties for reference
properties_df = pd.DataFrame(properties)
final_dataset = pd.concat([properties_df, embeddings_df], axis=1)

# Save the embeddings to a CSV file
output_path = 'Graph2Vec_Like_Embeddings.csv'
final_dataset.to_csv(output_path, index=False)

print(f"Graph2Vec-like embeddings saved to: {output_path}")
