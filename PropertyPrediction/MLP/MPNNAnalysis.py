# import pandas as pd
# import os
# import glob

# # Define the folder path containing the CSV files
# folder_path = 'C:/Users/eeo21/VSCodeProjects/MLForTribology/PropertyPrediction/GraphPrediction/MPNN/GridSearchResults'

# # Use glob to find all CSV files in the folder
# csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# # Initialize an empty list to store DataFrames
# dataframes = []

# # Iterate over the list of CSV files and read each one into a DataFrame
# for file in csv_files:
#     df = pd.read_csv(file)
#     dataframes.append(df)

# # Concatenate all DataFrames in the list into a single DataFrame
# combined_df = pd.concat(dataframes, ignore_index=True)

# # Save the combined DataFrame to a new CSV file (optional)
# combined_df.to_csv(os.path.join(folder_path, 'combined_grid_search_results.csv'), index=False)

# # Print a message indicating completion
# print(f"Combined {len(csv_files)} CSV files into a single DataFrame.")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import time
import matplotlib.pyplot as plt

# Load datasets
train_data_path = 'C:/Users/eeo21/VSCodeProjects/MLForTribology/Datasets/LargeTrainingDataset.csv'
test_data_path = 'C:/Users/eeo21/VSCodeProjects/MLForTribology/Datasets/EBDatasetSMILES.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

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

# Convert DataFrame to PyTorch Geometric Dataset
def create_dataset_from_dataframe(df):
    dataset = []
    for idx, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'])
        graph.y = torch.tensor([[row['visco@40C[cP]']]], dtype=torch.float)  # Ensure target has shape (N, 1)
        dataset.append(graph)
    return dataset

def create_dataset_from_dataframe_test(df):
    dataset = []
    for idx, row in df.iterrows():
        graph = smiles_to_graph(row['SMILES'])
        graph.y = torch.tensor([[row['Experimental_40C_Viscosity']]], dtype=torch.float)  # Ensure target has shape (N, 1)
        dataset.append(graph)
    return dataset

train_dataset = create_dataset_from_dataframe(train_df)
test_dataset = create_dataset_from_dataframe_test(test_df)

# Define the MPNN model with residual connections
class MPNN(MessagePassing):
    def __init__(self, hidden_dim, num_layers, dropout, activation):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation
        self.convs = torch.nn.ModuleList()
        self.convs.append(Linear(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(Linear(hidden_dim, hidden_dim))
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)
        self.dropout = dropout
        self.activation = activation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x_res = x
            x = self.activation(conv(x))
            x = self.propagate(edge_index, x=x)
            x = F.dropout(x, p=self.dropout, training=self.training) + x_res  # Residual connection
        x = global_mean_pool(x, data.batch)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def message(self, x_j):
        return x_j

# Retrieve the best model hyperparameters (assuming you've already identified the best model)
combined_results_path = 'C:/Users/eeo21/VSCodeProjects/MLForTribology/PropertyPrediction/GraphPrediction/MPNN/combined_grid_search_results.csv'
combined_df = pd.read_csv(combined_results_path)
best_model = combined_df.loc[combined_df['avg_test_loss'].idxmin()]

# Model parameters from best-performing model
hidden_dim = best_model['hidden_dim']
num_layers = best_model['num_layers']
learning_rate = best_model['learning_rate']
batch_size = best_model['batch_size']
dropout = best_model['dropout']
activation = getattr(F, best_model['activation'])  # Convert string to actual function reference

# Initialize the model with best hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNN(hidden_dim, num_layers, dropout, activation).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

# RMSE Loss function
def rmse_loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# Initialize a list to store the results
results = []

# Update your train_and_evaluate function to return training times and std
def train_and_evaluate(train_dataset, test_dataset, model, optimizer, rmse_loss, scheduler, epochs, batch_size):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    training_times = []

    for train_idx, val_idx in kf.split(train_dataset):
        train_subset = [train_dataset[i] for i in train_idx]
        val_subset = [train_dataset[i] for i in val_idx]

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        best_val_loss = float('inf')
        early_stopping_counter = 0
        start_time = time.time()  # Start time for training

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            # Training phase
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = rmse_loss(output, data.y)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output = model(data)
                    val_loss += rmse_loss(output, data.y).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), 'best_model_checkpoint.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 10:
                    print(f"Early stopping at epoch {epoch}")
                    break

        end_time = time.time()  # End time for training
        training_time = end_time - start_time  # Calculate training time
        training_times.append(training_time)

        results.append(best_val_loss)

    # Test the best model
    model.load_state_dict(torch.load('best_model_checkpoint.pth'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_rmse = 0

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            test_rmse += rmse_loss(output, data.y).item()

    test_rmse = test_rmse / len(test_loader)

    # Calculate standard deviation of validation losses
    std_val_loss = np.std(results)

    # Return results, standard deviation, and training times
    return results, std_val_loss, training_times, test_rmse

epochs = 100



# Train and evaluate the model
cv_results, std_val_loss, training_times, test_rmse = train_and_evaluate(
    train_dataset, test_dataset, model, optimizer, rmse_loss, scheduler, epochs, int(batch_size)
)

# Prepare data for saving to CSV
results_df = pd.DataFrame({
    'Training Data Size': [len(train_dataset)] * len(cv_results),
    'Cross-Validation RMSE': cv_results,
    'Standard Deviation': [std_val_loss] * len(cv_results),
    'Training Time': training_times
})

# Save the results to a CSV file
results_df.to_csv('cross_validation_results_Experimental_40C_GNN.csv', index=False)

