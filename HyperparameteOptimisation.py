# import os
# import pandas as pd
# import glob
# import re

# # Root directory containing subdirectories (Run_1, Run_2, ..., Run_X)
# root_dir = 'F:/PhD/HIGH_THROUGHPUT_STUDIES/generated_parameter_combinations'

# # Function to calculate average scores for elite performers
# def calculate_elite_averages(data, num_elite):
#     """Calculate average scores for elite performers."""
#     elite_data = data.head(num_elite)
#     target_columns = [
#         "DViscosity40C", "DViscosity100C", "HeatCapacity_40C",
#         "HeatCapacity_100C", "ThermalConductivity_40C", "ThermalConductivity_100C",
#         "DVI", "Toxicity", "SCScore", "TotalScore", "NichedScore"
#     ]
#     averages = {}
#     for column in target_columns:
#         averages[column] = round(elite_data[column].mean(), 3) if column in elite_data.columns else 0
#     return averages

# # Function to extract hyperparameters from GeneticAlgorithmML.py
# def extract_hyperparameters(genetic_algorithm_path):
#     """Extract relevant hyperparameters from the GeneticAlgorithmML.py file."""
#     with open(genetic_algorithm_path, 'r') as file:
#         content = file.read()
#     hyperparams = {}
#     for param in ["MaxGenerations", "MutationRate", "NumElite", "GenerationSize"]:
#         match = re.search(fr'{param}\s*=\s*([\d.]+)', content)
#         if match:
#             hyperparams[param] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
#         else:
#             raise ValueError(f"{param} not found in {genetic_algorithm_path}")
#     return hyperparams

# # Find all subdirectories (e.g., Run_1, Run_2, ..., Run_X)
# run_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# # Initialize a dictionary to store results for each target property
# target_columns = [
#         "DViscosity40C", "DViscosity100C", "HeatCapacity_40C",
#         "HeatCapacity_100C", "ThermalConductivity_40C", "ThermalConductivity_100C",
#         "DVI", "Toxicity", "SCScore", "TotalScore", "NichedScore"
# ]
# results_by_target = {col: [] for col in target_columns}

# # Process each run directory
# for run_dir in run_dirs:
#     # Extract hyperparameters from GeneticAlgorithmML.py
#     genetic_algorithm_path = os.path.join(run_dir, 'GeneticAlgorithmML.py')
#     hyperparams = extract_hyperparameters(genetic_algorithm_path)

#     # Create a dictionary to identify the hyperparameter combination
#     hyperparam_combination = {
#         "MaxGenerations": hyperparams["MaxGenerations"],
#         "MutationRate": hyperparams["MutationRate"],
#         "NumElite": hyperparams["NumElite"],
#         "GenerationSize": hyperparams["GenerationSize"]
#     }

#     # Find all CSV files with "generation" in their filename
#     csv_files = glob.glob(os.path.join(run_dir, '*generation*.csv'))

#     # Process each generation CSV
#     for file in csv_files:
#         # Extract the generation number from the filename
#         try:
#             generation_number = int([part for part in os.path.basename(file).split('_') if part.isdigit()][0])
#         except (IndexError, ValueError):
#             continue  # Skip files without valid generation numbers

#         # Read the CSV file
#         data = pd.read_csv(file)

#         # Calculate averages for the elite performers
#         averages = calculate_elite_averages(data, hyperparams['NumElite'])
#         averages["Generation"] = generation_number

#         # Append the hyperparameter combination and generation data
#         for target in target_columns:
#             results_by_target[target].append({
#                 "Generation": generation_number,
#                 "MaxGenerations": hyperparam_combination["MaxGenerations"],
#                 "MutationRate": hyperparam_combination["MutationRate"],
#                 "NumElite": hyperparam_combination["NumElite"],
#                 "GenerationSize": hyperparam_combination["GenerationSize"],
#                 target: averages[target]
#             })

# # Save each target property's results to a separate CSV
# output_dir = 'F:/PhD/HIGH_THROUGHPUT_STUDIES/generated_parameter_combinations/outputs'
# os.makedirs(output_dir, exist_ok=True)

# for target, results in results_by_target.items():
#     # Convert the list of results into a DataFrame
#     df = pd.DataFrame(results)
    
#     # Pivot the table so hyperparameters become column headers
#     pivot_df = df.pivot_table(
#         index="Generation",
#         columns=["MaxGenerations", "MutationRate", "NumElite", "GenerationSize"],
#         values=target
#     )

#     # Save to a CSV file
#     output_file = os.path.join(output_dir, f"{target}_Average_Scores.csv")
#     pivot_df.to_csv(output_file)

# print(f"Results saved to {output_dir}")


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data
# file_path = 'F:/PhD/HIGH_THROUGHPUT_STUDIES/generated_parameter_combinations/outputs/NichedScore_Average_Scores.csv'  # Replace with the path to your file
# data = pd.read_csv(file_path)

# # Inspect the data to understand its structure
# print(data.head())

# # Check if the necessary columns are present
# # Assuming 'Generation' column and hyperparameter combination columns exist
# if 'Generation' not in data.columns:
#     raise ValueError("The 'Generation' column is missing from the dataset.")

# # Identify hyperparameter combination columns (excluding 'Generation')
# hyperparameter_columns = [col for col in data.columns if col != 'Generation']

# # Plot the data
# plt.figure(figsize=(12, 8))
# for column in hyperparameter_columns:
#     plt.plot(data['Generation'], data[column], label=column, marker='o')

# # Customize the plot
# plt.title('Optimization of Metric Across Generations', fontsize=16)
# plt.xlabel('Generation', fontsize=14)
# plt.ylabel('NichedScore', fontsize=14)
# plt.legend(title='Hyperparameter Combinations', fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()

# # Show the plot
# plt.show()

# # Optional: Save the plot to a file
# # plt.savefig('/path/to/save/optimization_plot.png')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load and process the dataset
data = pd.read_csv('F:/PhD/HIGH_THROUGHPUT_STUDIES/generated_parameter_combinations/outputs/NichedScore_Average_Scores.csv')

# # Transpose and clean the data
# data = data.transpose()
# data.columns = data.iloc[0]  # Set first row as headers
# data = data[1:]  # Drop the header row
# data = data.reset_index(drop=True)
# data = data.rename(columns={"MutationRate": "Mutation_Rate",
#                              "NumElite": "Num_Elite",
#                              "GenerationSize": "Generation_Size"})
# data["Mutation_Rate"] = pd.to_numeric(data["Mutation_Rate"], errors='coerce')
# data["Num_Elite"] = pd.to_numeric(data["Num_Elite"], errors='coerce')
# data["Generation_Size"] = pd.to_numeric(data["Generation_Size"], errors='coerce')
# data = data.dropna(subset=["Mutation_Rate", "Num_Elite", "Generation_Size"])

# # Identify score columns and compute highest average score
# score_columns = data.columns.difference(["Mutation_Rate", "Num_Elite", "Generation_Size", "Generation"])
# data[score_columns] = data[score_columns].apply(pd.to_numeric, errors='coerce')
# aggregated_data = data.groupby(["Mutation_Rate", "Num_Elite", "Generation_Size"])[score_columns].mean().max(axis=1).reset_index(name="Highest_Average_Score")

# # Prepare data for 3D surface plot without aggregation of Generation_Size
# X = aggregated_data["Mutation_Rate"].values
# Y = aggregated_data["Num_Elite"].values
# Z = aggregated_data["Generation_Size"].values
# scores = aggregated_data["Highest_Average_Score"].values

# # Create grid for interpolation
# xi = np.linspace(X.min(), X.max(), 1000)
# yi = np.linspace(Y.min(), Y.max(), 1000)
# xi, yi = np.meshgrid(xi, yi)
# zi_gen_size = griddata((X, Y), Z, (xi, yi), method='cubic')
# zi_scores = griddata((X, Y), scores, (xi, yi), method='cubic')

# # Rescale the scores to map the full range from 0.86 to max_score across the color map
# max_score = 0.915
# min_score = 0.8
# normalized_full_range_scores = (zi_scores - min_score) / (max_score - min_score)  # Rescale to [0, 1]
# normalized_full_range_scores = np.clip(normalized_full_range_scores, 0.1, 1)

# # Plotting the 3D surface plot with full-range adjusted heatmap
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Surface plot with adjusted color map for the full range
# surf_full_range = ax.plot_surface(
#     xi, yi, zi_gen_size, facecolors=plt.cm.jet(normalized_full_range_scores), edgecolor='none'
# )

# # Add color bar for the full-range adjusted heatmap
# mappable_full_range = plt.cm.ScalarMappable(cmap='jet')
# mappable_full_range.set_array(np.linspace(min_score, max_score, 1000))  # Color bar reflects the adjusted range
# fig.colorbar(mappable_full_range, ax=ax, shrink=0.5, aspect=10, label=f"Highest Average Score")

# # Axis labels
# ax.set_xlabel("Mutation Rate")
# ax.set_ylabel("Num Elite")
# ax.set_zlabel("Generation Size")

# # Title
# plt.title(f"3D Surface Plot: Hyperparameter Optimisation")
# plt.savefig(f'F:/PhD/HIGH_THROUGHPUT_STUDIES/generated_parameter_combinations/outputs/Optimisation.png')
# plt.show()

# Reload and process the dataset
data = pd.read_csv('F:/PhD/HIGH_THROUGHPUT_STUDIES/generated_parameter_combinations/outputs/NichedScore_Average_Scores.csv').transpose()
data.columns = data.iloc[0]  # First row as header
data = data[1:]  # Remove header row
data.reset_index(drop=True, inplace=True)
data = data.rename(columns={"MutationRate": "Mutation_Rate",
                             "NumElite": "Num_Elite",
                             "GenerationSize": "Generation_Size"})
data["Mutation_Rate"] = pd.to_numeric(data["Mutation_Rate"], errors='coerce')
data["Num_Elite"] = pd.to_numeric(data["Num_Elite"], errors='coerce')
data["Generation_Size"] = pd.to_numeric(data["Generation_Size"], errors='coerce')

# Find the generation of the highest score for each combination
score_columns = data.columns[~data.columns.isin(["Mutation_Rate", "Num_Elite", "Generation_Size", "Generation"])]
data[score_columns] = data[score_columns].apply(pd.to_numeric, errors='coerce')
data["Best_Generation"] = data[score_columns].idxmax(axis=1).astype(float)

# Prepare data for the plot
X = data["Mutation_Rate"].values
Y = data["Num_Elite"].values
Z = data["Generation_Size"].values
generations = data["Best_Generation"].values

# Create grid for interpolation
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi_gen_size = griddata((X, Y), Z, (xi, yi), method='cubic')
zi_generations = griddata((X, Y), generations, (xi, yi), method='cubic')

# Normalize generations for heatmap
normalized_generations = (zi_generations - zi_generations.min()) / (zi_generations.max() - zi_generations.min())

# Plot the 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot with Best Generation heatmap
surf_generations = ax.plot_surface(
    xi, yi, zi_gen_size, facecolors=plt.cm.jet(1 - normalized_generations), edgecolor='none'
)

# Add color bar
mappable_generations = plt.cm.ScalarMappable(cmap='jet')
mappable_generations.set_array(generations)
fig.colorbar(mappable_generations, ax=ax, shrink=0.5, aspect=10, label="Best Generation (Lower is Better)")

# Axis labels
ax.set_xlabel("Mutation Rate")
ax.set_ylabel("Num Elite")
ax.set_zlabel("Generation Size")

# Title
plt.title("3D Surface Plot: Best Generation Heatmap")

plt.show()