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
import time

RDS = True

if RDS:
    df = pd.read_csv('/lustre/scratch/mmm1058/MLForTribology/Datasets/GridSearchDataset.csv')
else:
    df = pd.read_csv('Datasets/GridSearchDataset.csv')

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
    def __init__(self, hidden_dim, num_layers, dropout, activation):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(1, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x_res = x
            x = self.activation(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training) + x_res  # Residual connection
        x = global_mean_pool(x, data.batch)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameter grid
hidden_dims = [16, 32, 64, 128, 256]
num_layers = [1, 2, 3, 4, 5]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [8, 16, 32, 64, 128]
dropout_rates = [0.0]
activations = [F.relu, F.leaky_relu, torch.tanh, torch.sigmoid, F.elu]
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']

# Initialize results file
results_file = 'graphsage_grid_search_results.csv'
if not os.path.exists(results_file):
    results_df = pd.DataFrame(columns=['hidden_dim', 'num_layers', 'learning_rate', 'batch_size', 'dropout', 'activation', 'optimizer', 'avg_test_loss', 'training_time'])
    results_df.to_csv(results_file, index=False)

# Perform grid search with K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for hidden_dim, num_layer, lr, batch_size, dropout, activation, opt in itertools.product(hidden_dims, num_layers, learning_rates, batch_sizes, dropout_rates, activations, optimizers):
    fold_results = []
    fold_times = []
    for train_index, test_index in kf.split(data_list):
        train_data = [data_list[i] for i in train_index]
        test_data = [data_list[i] for i in test_index]

        print(len(train_data))
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GraphSAGE(hidden_dim, num_layer, dropout, activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) if opt == 'adam' else \
                    torch.optim.SGD(model.parameters(), lr=lr) if opt == 'sgd' else \
                    torch.optim.RMSprop(model.parameters(), lr=lr) if opt == 'rmsprop' else \
                    torch.optim.Adagrad(model.parameters(), lr=lr)
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
        start_time = time.time()
        
        for epoch in tqdm(range(100), desc=f'Training Epochs for HD: {hidden_dim}, NL: {num_layer}, LR: {lr}, BS: {batch_size}, DR: {dropout}, ACT: {activation.__name__}, OPT: {opt}'):
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
        
        end_time = time.time()
        training_time = end_time - start_time
        
        test_loss = test(test_loader)
        fold_results.append(test_loss)
        fold_times.append(training_time)
    
    avg_test_loss = np.mean(fold_results)
    avg_training_time = np.mean(fold_times)
    
    result = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layer,
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout': dropout,
        'activation': activation.__name__,
        'optimizer': opt,
        'avg_test_loss': avg_test_loss,
        'training_time': avg_training_time
    }
    
    results_df = pd.DataFrame([result])
    results_df.to_csv(results_file, mode='a', header=False, index=False)
    print(f'Hidden Dim: {hidden_dim}, Num Layers: {num_layer}, LR: {lr}, Batch Size: {batch_size}, Dropout: {dropout}, Activation: {activation.__name__}, Optimizer: {opt}, Avg Test Loss: {avg_test_loss}, Training Time: {avg_training_time}')
