import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.model_selection import KFold
import itertools
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the dataset
file_path = 'Datasets/FinalDataset.csv'
df = pd.read_csv(file_path)

# Function to convert SMILES to molecular graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    
    # Get bonds
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    
    # Convert to tensor
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# Convert SMILES strings to graph data
data_list = []
for i, row in df.iterrows():
    graph = smiles_to_graph(row['smiles'])
    graph.y = torch.tensor([[row['visco@40C[cP]']]], dtype=torch.float)  # Ensure target has shape (N, 1)
    data_list.append(graph)

# Define the GraphSAGE model with residual connections and dropout
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(1, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x_res = x
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training) + x_res  # Residual connection
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameter grid
hidden_dims = [64, 128]
num_layers = [2, 3]
learning_rates = [0.01, 0.001]
batch_sizes = [32, 64]
dropout_rates = [0.2, 0.3]

# Initialize results file
results_file = 'graphsage_grid_search_results.csv'
if not os.path.exists(results_file):
    results_df = pd.DataFrame(columns=['hidden_dim', 'num_layers', 'learning_rate', 'batch_size', 'dropout', 'avg_test_loss'])
    results_df.to_csv(results_file, index=False)

# Perform grid search with K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for hidden_dim, num_layer, lr, batch_size, dropout in itertools.product(hidden_dims, num_layers, learning_rates, batch_sizes, dropout_rates):
    fold_results = []
    for train_index, test_index in kf.split(data_list):
        train_data = [data_list[i] for i in train_index[:int(0.2 * len(train_index))]]
        test_data = [data_list[i] for i in test_index]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GraphSAGE(hidden_dim, num_layer, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
        criterion = torch.nn.MSELoss()
        
        def train():
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        
        def test(loader):
            model.eval()
            error = 0
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data)
                    error += criterion(out, data.y).item()
            return error / len(loader)
        
        early_stopping_patience = 20
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(100), desc=f'Training Epochs for HD: {hidden_dim}, NL: {num_layer}, LR: {lr}, BS: {batch_size}, DR: {dropout}'):
            train()
            test_loss = test(test_loader)
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation loss.")
                break
        
        test_loss = test(test_loader)
        fold_results.append(test_loss)
    
    avg_test_loss = np.mean(fold_results)
    result = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layer,
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout': dropout,
        'avg_test_loss': avg_test_loss
    }
    results_df = pd.DataFrame([result])
    results_df.to_csv(results_file, mode='a', header=False, index=False)
    print(f'Hidden Dim: {hidden_dim}, Num Layers: {num_layer}, LR: {lr}, Batch Size: {batch_size}, Dropout: {dropout}, Avg Test Loss: {avg_test_loss}')
