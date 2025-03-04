import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = "CyclicBenchmarkingResults.xlsx"
xls = pd.ExcelFile(file_path)

# Load the data from the first sheet
df = pd.read_excel(xls, sheet_name='Sheet1')

# Extract unique molecules (removing temperature suffix)
df["Molecule"] = df["Unnamed: 0"].apply(lambda x: "_".join(x.split("_")[:-1]))

# Define temperature points
temperatures = [313, 373]

# Define method names including Experimental
methods = ['LOPLS_GreenKubo', 'LOPLS_Einstein', 'AMBER_GreenKubo', 'AMBER_Einstein',
           'COMPASS_Einstein', 'COMPASS_GreenKubo', 'SCIPCFF_GreenKubo', 'SCIPCFF_Einstein',
           'ML Prediction', 'Experimental']

# Define standard deviation mappings (excluding Experimental since it has no uncertainty)
std_columns = {
    'LOPLS_GreenKubo': 'LOPLS_GreenKubo_Std', 
    'LOPLS_Einstein': 'LOPLS_Einstein_Std', 
    'AMBER_GreenKubo': 'AMBER_GreenKubo_Std', 
    'AMBER_Einstein': 'AMBER_Einstein_Std', 
    'COMPASS_Einstein': 'COMPASS_Einstein_Std', 
    'COMPASS_GreenKubo': 'COMPASS_GreenKubo_Std', 
    'SCIPCFF_GreenKubo': 'SCIPCFF_GreenKubo_Std', 
    'SCIPCFF_Einstein': 'SCIPCFF_Einstein_Std'
}

# Define a color map for each method, ensuring Experimental has a distinct color
color_map = plt.get_cmap("tab10")
method_colors = {method: color_map(i) for i, method in enumerate(methods)}
method_colors["Experimental"] = "black"  # Make Experimental stand out

# Plot the performance of different methods for each molecule at both temperatures
for molecule in df["Molecule"].unique():
    plt.figure(figsize=(12, 6))
    
    # Filter data for the molecule
    mol_data = df[df["Molecule"] == molecule]
    
    # Define positions for bars with a gap between the two temperatures
    x_labels = [f"{temp}K" for temp in temperatures]
    x = np.array([0, 1.25])  # Increase gap between temperature groups
    width = 0.1  # Width of each bar

    # Plot bars for each method
    for i, method in enumerate(methods):
        stds = mol_data[std_columns[method]].values if method in std_columns else np.zeros(len(mol_data))

        plt.bar(x + i * width - (len(methods) / 2) * width, 
                mol_data[method].values, 
                width=width, 
                label=method,  # Ensure legend appears for each molecule
                color=method_colors[method],  # Assign color based on method
                yerr=stds, 
                capsize=4, 
                error_kw={'elinewidth': 1, 'capsize': 5, 'capthick': 1})

    # Labels and titles
    plt.xticks(x, x_labels, fontsize=12)
    plt.ylabel("Viscosity (cP)", fontsize=12)
    plt.title(f"Comparison of MD Simulations, ML Prediction, and Experimental for {molecule}", fontsize=14)
    
    # Ensure the legend appears for every plot
    plt.legend(fontsize=10, loc="upper right")
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
