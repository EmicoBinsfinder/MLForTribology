import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch_geometric.nn import MessagePassing, global_mean_pool
from captum.attr import IntegratedGradients
from sklearn.model_selection import KFold

# Load dataset from JSON
with open("MolecularGraphDataset.json", "r") as f:
    dataset = json.load(f)

# Define MPNN model
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')  # Sum aggregation
        self.edge_mlp = nn.Linear(1, out_channels)  # Transform edge attributes
        self.node_mlp = nn.Linear(in_channels + out_channels, out_channels)  # Update node embeddings

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1, 1)  # Ensure 2D shape

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            edge_attr = torch.zeros((x_j.shape[0], 1), device=x_j.device)  # Default zero if missing

        edge_embedding = self.edge_mlp(edge_attr)  # Transform edge features
        return torch.cat([x_j, edge_embedding], dim=1)  # Concatenate with node features

    def update(self, aggr_out):
        return torch.relu(self.node_mlp(aggr_out))  # Update node embeddings

class MPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super(MPNN, self).__init__()
        self.layers = nn.ModuleList([
            MPNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if edge_attr is not None:
            edge_attr = edge_attr.view(-1, 1)  # Ensure correct shape

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.dropout(x)  # Apply dropout for regularization
        x = global_mean_pool(x, data.batch)  # Aggregate node embeddings
        return self.fc(x)

# Training Hyperparameters
HYPERPARAMS = {
    "hidden_dim": 32,
    "num_layers": 4,
    "learning_rate": 0.01,
    "batch_size": 8,
    "dropout": 0,
    "activation": "leaky_relu",
    "optimizer": "rmsprop"
}

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of target properties
target_properties = [
    "Thermal_Conductivity_40C", "Thermal_Conductivity_100C",
    "Density_40C", "Density_100C",
    "Viscosity_40C", "Viscosity_100C",
    "Heat_Capacity_40C", "Heat_Capacity_100C"
]

for target in target_properties:
    
    print(f"\nTraining for {target}...")

    # Prepare dataset for the target property
    graph_data = []
    for item in dataset:
        graph = item["graph"]
        properties = graph["global_properties"]

        if target not in properties or properties[target] is None or np.isnan(properties[target]):
            continue  # Skip molecules missing this property

        x = torch.tensor([list(node.values()) for node in graph["nodes"]], dtype=torch.float)  # No normalization
        edge_index = torch.tensor([[edge["begin_atom_idx"], edge["end_atom_idx"]] for edge in graph["edges"]], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([edge["bond_length"] for edge in graph["edges"]], dtype=torch.float).view(-1, 1)  # No normalization

        y_value = torch.tensor([properties[target]], dtype=torch.float).view(-1, 1)  # Keep target unchanged

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_value)
        graph_data.append(data)

    if len(graph_data) == 0:
        print(f"Skipping {target} - No valid data points.")
        continue

    start_time = time.time()
    fold_mse = []
    best_val_loss = float('inf')
    best_model_path = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(graph_data)):
        train_set = [graph_data[i] for i in train_idx]
        val_set = [graph_data[i] for i in val_idx]

        train_loader = DataLoader(train_set, batch_size=HYPERPARAMS["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=HYPERPARAMS["batch_size"])

        model = MPNN(input_dim=train_set[0].x.shape[1], hidden_dim=HYPERPARAMS["hidden_dim"], num_layers=HYPERPARAMS["num_layers"]).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=HYPERPARAMS["learning_rate"])
        loss_fn = nn.MSELoss()

        patience = 10
        patience_counter = 0

        for epoch in range(100):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = loss_fn(output, batch.y.view(-1, 1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step()

            model.eval()
            val_loss = sum(loss_fn(model(batch.to(device)), batch.y.view(-1, 1)).item() for batch in val_loader) / len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f"mpnn_{target.replace(' ', '_')}.pt"
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        fold_mse.append(best_val_loss)

    mean_mse = np.mean(fold_mse)
    best_mse = np.min(fold_mse)
    std_mse = np.std(fold_mse)
    training_time = time.time() - start_time

    results.append([target, mean_mse, best_mse, std_mse, training_time])

pd.DataFrame(results, columns=["Property", "Mean MSE", "Best MSE", "MSE Std Dev", "Training Time (s)"]).to_csv("mpnn_training_results.csv", index=False)

print("\nTraining completed. Best models and results saved.")


# Feature Importance Analysis (GraphSHAP & IG)
for target in ["Thermal Conductivity 40C", "Density 40C", "Viscosity 40C"]:  # Limit to a few targets for analysis
    model = MPNN(input_dim=graph_data[0].x.shape[1], hidden_dim=HYPERPARAMS["hidden_dim"], num_layers=HYPERPARAMS["num_layers"]).to(device)
    model.load_state_dict(torch.load(f"mpnn_{target.replace(' ', '_')}.pt"))
    model.eval()

    valid_data_loader = DataLoader(graph_data, batch_size=8, shuffle=False)
    
    # SHAP Analysis
    explainer = shap.Explainer(model, torch.cat([d.x for d in graph_data]).to(device))
    shap_values = explainer(torch.cat([d.x for d in graph_data]).to(device))

    shap_values_np = shap_values.values.cpu().detach().numpy()
    pd.DataFrame(shap_values_np.mean(axis=0), columns=["SHAP Importance"]).to_csv(f"shap_importance_{target.replace(' ', '_')}.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(shap_values_np.mean(axis=0))), shap_values_np.mean(axis=0))
    plt.xlabel("Feature Index")
    plt.ylabel("SHAP Value")
    plt.title(f"SHAP Feature Importance for {target}")
    plt.savefig(f"shap_importance_{target.replace(' ', '_')}.png")
    plt.show()

    # Integrated Gradients
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(graph_data[0].x).to(device)
    ig_attr = ig.attribute(torch.cat([d.x for d in graph_data]).to(device), baselines=baseline)
    
    ig_attr_np = ig_attr.cpu().detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(ig_attr_np.mean(axis=0))), ig_attr_np.mean(axis=0))
    plt.xlabel("Feature Index")
    plt.ylabel("Integrated Gradients Score")
    plt.title(f"IG Feature Importance for {target}")
    plt.savefig(f"ig_importance_{target.replace(' ', '_')}.png")
    plt.show()
