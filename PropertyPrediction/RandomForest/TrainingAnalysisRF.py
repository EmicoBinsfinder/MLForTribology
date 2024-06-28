import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

# Step 1: Read the CSV file
file_path = 'PropertyPrediction/RandomForest/RFResults.csv'
data = pd.read_csv(file_path)

# Step 2: Extract hyperparameters
data['Params'] = data['Params'].apply(ast.literal_eval)
hyperparameters = pd.json_normalize(data['Params'])
data = pd.concat([data, hyperparameters], axis=1).drop(columns=['Params'])

# Convert hyperparameters to numeric, handling errors by setting invalid parsing to NaN
hyperparameters = hyperparameters.apply(pd.to_numeric, errors='coerce')

# Drop columns or rows with NaN values if necessary
hyperparameters = hyperparameters.dropna(axis=1, how='all')  # Drop columns where all values are NaN
hyperparameters = hyperparameters.dropna(axis=0, how='any')  # Drop rows with any NaN values

# Align data with the cleaned hyperparameters
data = data.loc[hyperparameters.index]

# Step 3: Calculate the correlation between hyperparameters and Validation MSE
correlation = hyperparameters.corrwith(data['Validation MSE']).abs()
most_impactful = correlation.nlargest(2).index.tolist()

# Step 4: Identify the best-performing set of hyperparameters
best_index = data['Validation MSE'].idxmin()
best_hyperparameters = data.loc[best_index, most_impactful]
best_mse = data.loc[best_index, 'Validation MSE']

# Print the best-performing set of hyperparameters
print(f"Best-performing hyperparameters:")
for param in most_impactful:
    print(f"{param}: {best_hyperparameters[param]}")
print(f"Validation MSE: {best_mse}")

# Step 5: Create a pivot table for the heatmap
pivot_table = data.pivot_table(values='Validation MSE', index=most_impactful[0], columns=most_impactful[1])

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="jet", cbar_kws={'label': 'Validation MSE'})
plt.title(f'Heatmap of {most_impactful[0]} and {most_impactful[1]} vs Validation MSE')
plt.xlabel(most_impactful[1])
plt.ylabel(most_impactful[0])
plt.show()
