import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_smiles
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
import pandas as pd
import pickle

# Load the dataset
file_path = 'Datasets/FinalDataset.csv'  # Update with the correct path
data = pd.read_csv(file_path)

# Extract SMILES strings and target viscosity values
smiles = data['smiles']
viscosity = data['WhackVisc']

# Split the data: 20% for training, 80% for testing
train_smiles, test_smiles, train_viscosity, test_viscosity = train_test_split(smiles, viscosity, test_size=0.8, random_state=42)

# Define the MPNN Model
class MPNNModel(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(MPNNModel, self).__init__()
        self.conv1 = MessagePassing(aggr='add')
        self.conv2 = MessagePassing(aggr='add')
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

# Custom MessagePassing implementation (simplified example)
class SimpleMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SimpleMessagePassing, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, num_edges]
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j has shape [num_edges, in_channels]
        return self.linear(x_j)
    
    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]
        return aggr_out

# Convert the SMILES strings to molecular graphs
def smiles_to_graph(smiles):
    return from_smiles(smiles)

train_graphs = [smiles_to_graph(s) for s in train_smiles]
test_graphs = [smiles_to_graph(s) for s in test_smiles]

# Define hyperparameter grid
param_grid = {
    'hidden_channels': [64, 128],
    'out_channels': [64, 128],
    'learning_rate': [0.001, 0.01],
    'batch_size': [16, 32]
}

# Prepare the CSV file for incremental saving
results_file_path = 'grid_search_results.csv'
results_columns = ['hidden_channels', 'out_channels', 'learning_rate', 'batch_size', 'fold', 'loss']
results_df = pd.DataFrame(columns=results_columns)
results_df.to_csv(results_file_path, index=False)

# Perform grid search with 5-fold cross-validation
kf = KFold(n_splits=5)
results = []

# Perform grid search
for params in ParameterGrid(param_grid):
    fold_results = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(train_graphs)):
        # Prepare training and validation data
        train_data = [train_graphs[i] for i in train_index]
        val_data = [train_graphs[i] for i in val_index]
        
        train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=params['batch_size'])
        
        # Initialize model, loss function, and optimizer
        model = MPNNModel(params['hidden_channels'], params['out_channels'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        for epoch in range(100):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
        
        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                val_losses.append(loss.item())
        
        fold_result = {
            'hidden_channels': params['hidden_channels'],
            'out_channels': params['out_channels'],
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'fold': fold,
            'loss': np.mean(val_losses)
        }
        fold_results.append(fold_result)
        
        # Append the result to the CSV file incrementally
        results_df = pd.DataFrame([fold_result])
        results_df.to_csv(results_file_path, mode='a', header=False, index=False)
    
    # Save results for current set of hyperparameters
    results.append({'params': params, 'loss': np.mean([r['loss'] for r in fold_results])})

# Save the overall results
results_df = pd.DataFrame(results)
results_df.to_csv('overall_grid_search_results.csv', index=False)

# Display the best hyperparameters
best_params = results_df.loc[results_df['loss'].idxmin()]
print(best_params)
