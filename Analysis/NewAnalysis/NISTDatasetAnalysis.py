'''
Date 19th November 2024

Author: Egheosa Ogbomo

Script to analyse composition of NIST Webbook dataset

You are a PhD level researcher capable of performing analysis of datasets such as that in the 
uploaded 

Please provide the code to do the following analysis of the uploaded dataset.

For each point below create a new section in the python code

We want:

- The total number of datapoints for each of the four thermophysical properties (Thermal conductivity,
density, viscosity, heat capacity)
- Create a table showing the number of molecule types (Ester, ether, Paraffin, Cyclic Paraffin, Aromatic etc)
- Number of datapoints for each property at each temperature
- Spread by number of heavy atoms as a bar chart, to do this extract the number of heavy atoms from the SMILES strings



- Create a new column
- Scatter graph showing spread of viscosity vs number of heavy atoms (same for thermal conductivity)
- Number of values for each analytical model for each property
- 5 examples showing the accuracy of each models predictions for each property


What I need to do to the dataset:
- Split dataset into smaller dataset such that it has Density, Heat Capacity, Viscosity and Thermal 
Conductivity values at each temperature (40 and 100), and 

- Further split into training and test sets for all of the properties 80:10:10 (random)

- Pick 5 examples of molecules that I can get actual data for each evaluation model and compare these values
to experimental/literature values

'''

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import seaborn as sns
from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Load the dataset
file_path = 'C:/Users/eeo21/Desktop/Datasets/NISTCyclicMoleculeDataset_LogScale.csv'
dataset = pd.read_csv(file_path)

# Define properties and corresponding columns for both temperatures
properties = {
    "Thermal Conductivity 40C": "Thermal Conductivity_40C",
    "Thermal Conductivity 100C": "Thermal Conductivity_100C",
    "Density 40C": "Density_40C (g/cm^3)",
    "Density 100C": "Density_100C (g/cm^3)",
    "Viscosity 40C": "Log_Viscosity_40C",
    "Viscosity 100C": "Log_Viscosity_100C",
    "Heat Capacity 40C": "Log_Heat_Capacity_40C",
    "Heat Capacity 100C": "Log_Heat_Capacity_100C",
}

# Get all RDKit molecular descriptors
descriptor_names = [desc[0] for desc in Chem.Descriptors._descList]
calculator = MolecularDescriptorCalculator(descriptor_names)

# Function to calculate molecular descriptors for a SMILES string
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return calculator.CalcDescriptors(mol)
    except:
        pass
    return [None] * len(descriptor_names)

# Loop through each property, calculate descriptors, and save
for property_name, column in properties.items():
    # Drop rows with missing values in the target column
    property_data = dataset.dropna(subset=[column])

    # Rename the target property column to 'Property_Temp'
    renamed_column = f"{property_name.replace(' ', '_')}"
    property_data = property_data.rename(columns={column: renamed_column})

    # Keep only SMILES and target property
    property_data = property_data[["SMILES", renamed_column]]
    
    # Calculate molecular descriptors
    descriptor_values = property_data["SMILES"].apply(calculate_descriptors)
    descriptors_df = pd.DataFrame(descriptor_values.tolist(), columns=descriptor_names)

    # Remove columns with only zero values
    descriptors_df = descriptors_df.loc[:, (descriptors_df != 0).any(axis=0)]

    # Combine SMILES, descriptors, and target property
    combined_data = pd.concat([property_data.reset_index(drop=True), descriptors_df], axis=1)

    # Perform an 85/15 split
    train_data, test_data = train_test_split(combined_data, test_size=0.15, random_state=42)

    # Save the datasets
    train_file = f"C:/Users/eeo21/Desktop/Datasets/{property_name.replace(' ', '_')}_Train_Descriptors.csv"
    test_file = f"C:/Users/eeo21/Desktop/Datasets/{property_name.replace(' ', '_')}_Test_Descriptors.csv"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Training set with descriptors saved to: {train_file}")
    print(f"Test set with descriptors saved to: {test_file}")

# Loop through each property, split data, and save
for property_name, column in properties.items():
    # Drop rows with missing values in the target column
    property_data = dataset.dropna(subset=[column])

    # Rename the target property column to 'Property_Temp'
    renamed_column = f"{property_name.replace(' ', '_')}"
    property_data = property_data.rename(columns={column: renamed_column})

    # Keep only the SMILES string and renamed target column
    property_data = property_data[["SMILES", renamed_column]]

    # Split into features (X) and target (y)
    X = property_data.drop(columns=[renamed_column])
    y = property_data[renamed_column]

    # Perform an 85/15 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Combine X and y for train and test sets
    train_data = X_train.copy()
    train_data[renamed_column] = y_train

    test_data = X_test.copy()
    test_data[renamed_column] = y_test

    # Save the datasets
    train_file = f"C:/Users/eeo21/Desktop/Datasets/{renamed_column}_Train.csv"
    test_file = f"C:/Users/eeo21/Desktop/Datasets/{renamed_column}_Test.csv"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Training set saved to: {train_file}")
    print(f"Test set saved to: {test_file}")

# # Extract prediction columns from the dataset
# prediction_columns = [
#     "Viscosity Prediction",
#     "Density Prediction",
#     "Thermal Conductivity Prediction",
#     "Heat Capacity Prediction",
# ]

# # Prepare a list to store the results
# prediction_summary = []

# # Count occurrences of each prediction type for each property
# for col in prediction_columns:
#     property_name = col.replace(" Prediction", "")
#     prediction_counts = dataset[col].value_counts()
#     for prediction_type, count in prediction_counts.items():
#         prediction_summary.append({
#             "Property": property_name,
#             "Prediction Type": prediction_type,
#             "Count": count
#         })

# # Convert the results into a DataFrame
# prediction_summary_df = pd.DataFrame(prediction_summary)

# # Display the results
# print(prediction_summary_df)


# # Ensure density values are in g/cm^3 and convert to kg/m^3
# dataset["Density_40C (kg/m^3)"] = dataset["Density_40C (g/cm^3)"] * 1000
# dataset["Density_100C (kg/m^3)"] = dataset["Density_100C (g/cm^3)"] * 1000

# # Initialize new columns for dynamic and kinematic viscosities (if conversions are needed)
# dataset["Kinematic Viscosity 40°C (cSt)"] = None
# dataset["Kinematic Viscosity 100°C (cSt)"] = None
# dataset["Dynamic Viscosity 40°C (cP)"] = None
# dataset["Dynamic Viscosity 100°C (cP)"] = None

# # Perform the conversions
# for index, row in dataset.iterrows():
#     # For 40°C
#     if pd.notnull(row["Density_40C (kg/m^3)"]) and pd.notnull(row["Experimental_40C_Viscosity"]):
#         if row["Dynamic Or Kinematic"].lower() == "dynamic":
#             # Convert dynamic (cP) to kinematic (cSt)
#             dynamic_visc = row["Experimental_40C_Viscosity"]  # cP
#             kinematic_visc = dynamic_visc / row["Density_40C (g/cm^3)"]  # cSt
#             dataset.at[index, "Dynamic Viscosity 40°C (cP)"] = dynamic_visc
#             dataset.at[index, "Kinematic Viscosity 40°C (cSt)"] = kinematic_visc
#         elif row["Dynamic Or Kinematic"].lower() == "kinematic":
#             # Convert kinematic (cSt) to dynamic (cP)
#             kinematic_visc = row["Experimental_40C_Viscosity"]  # cSt
#             dynamic_visc = kinematic_visc * row["Density_40C (g/cm^3)"]  # cP
#             dataset.at[index, "Kinematic Viscosity 40°C (cSt)"] = kinematic_visc
#             dataset.at[index, "Dynamic Viscosity 40°C (cP)"] = dynamic_visc

#     # For 100°C
#     if pd.notnull(row["Density_100C (kg/m^3)"]) and pd.notnull(row["Experimental_100C_Viscosity"]):
#         if row["Dynamic Or Kinematic"].lower() == "dynamic":
#             # Convert dynamic (cP) to kinematic (cSt)
#             dynamic_visc = row["Experimental_100C_Viscosity"]  # cP
#             kinematic_visc = dynamic_visc / row["Density_100C (g/cm^3)"]  # cSt
#             dataset.at[index, "Dynamic Viscosity 100°C (cP)"] = dynamic_visc
#             dataset.at[index, "Kinematic Viscosity 100°C (cSt)"] = kinematic_visc
#         elif row["Dynamic Or Kinematic"].lower() == "kinematic":
#             # Convert kinematic (cSt) to dynamic (cP)
#             kinematic_visc = row["Experimental_100C_Viscosity"]  # cSt
#             dynamic_visc = kinematic_visc * row["Density_100C (g/cm^3)"]  # cP
#             dataset.at[index, "Kinematic Viscosity 100°C (cSt)"] = kinematic_visc
#             dataset.at[index, "Dynamic Viscosity 100°C (cP)"] = dynamic_visc

# # Save the updated dataset to a new file
# new_file_path = 'C:/Users/eeo21/Desktop/Datasets/CyclicMoleculeBenchmarkingDataset_with_ViscosityColumns_AllTemperatures.csv'
# dataset.to_csv(new_file_path, index=False)

# print(f"Dataset saved with new viscosity columns for both temperatures to {new_file_path}.")


# ##### Section 1: Total Number of Datapoints for Each Property
# properties = {
#     "Thermal Conductivity": ["Thermal Conductivity_40C", "Thermal Conductivity_100C"],
#     "Density": ["Density_40C (g/cm^3)", "Density_100C (g/cm^3)"],
#     "Viscosity": ["Experimental_40C_Viscosity", "Experimental_100C_Viscosity"],
#     "Heat Capacity": [
#         "Heat Capacity (Constant Pressure) 40C (J/K/Mol)",
#         "Heat Capacity (Constant Pressure) 100C (J/K/Mol)",
#     ],
# }

# property_counts = {
#     prop: dataset[cols].notnull().sum().sum() for prop, cols in properties.items()
# }
# property_counts_df = pd.DataFrame(property_counts.items(), columns=["Property", "Total Datapoints"])
# print(property_counts_df)

# ##### Section 2: Molecule Types Frequency Table
# molecule_type_counts = dataset["Type"].value_counts().reset_index()
# molecule_type_counts.columns = ["Molecule Type", "Count"]
# print(molecule_type_counts)

# ##### Section 4: Spread by Number of Heavy Atoms
# def count_heavy_atoms(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
#     except:
#         return None

# Add a new column for heavy atom counts
# dataset["Heavy Atoms"] = dataset["SMILES"].apply(count_heavy_atoms)

# # Save the updated dataset to a new file
# new_file_path = 'C:/Users/eeo21/Desktop/Datasets/CyclicMoleculeBenchmarkingDataset_with_HeavyAtoms.csv'
# dataset.to_csv(new_file_path, index=False)

# print(f"Dataset saved with heavy atom counts added to {new_file_path}.")

# dataset["Heavy Atoms"] = dataset["SMILES"].apply(count_heavy_atoms)
# heavy_atoms_counts = dataset["Heavy Atoms"].value_counts().sort_index()

# # Bar Chart for Heavy Atoms
# plt.figure(figsize=(10, 6))
# plt.bar(heavy_atoms_counts.index, heavy_atoms_counts.values, width=0.8, color='b')
# plt.title("Distribution of Molecules by Number of Heavy Atoms")
# plt.xlabel("Number of Heavy Atoms")
# plt.ylabel("Frequency")
# plt.grid(axis="y")
# plt.savefig('C:/Users/eeo21/Desktop/GeneticAlgoPaperImages/SpreadbyHeavyAtomsNum.png')
# plt.show()


# # Define the properties to plot
# properties_to_plot = {
#     "Thermal Conductivity 40°C [W/m/K]" : "Thermal Conductivity_40C",
#     "Thermal Conductivity 100°C [W/m/K]": "Thermal Conductivity_100C",
#     "Density 40°C (g/cm³)": "Density_40C (g/cm^3)",
#     "Density 100°C (g/cm³)": "Density_100C (g/cm^3)",
#     "Viscosity 40°C [mPa.s]": "Experimental_40C_Viscosity",
#     "Viscosity 100°C [mPa.s]": "Experimental_100C_Viscosity",
#     "Heat Capacity 40°C (J/K/Mol)": "Heat Capacity (Constant Pressure) 40C (J/K/Mol)",
#     "Heat Capacity 100°C (J/K/Mol)": "Heat Capacity (Constant Pressure) 100C (J/K/Mol)",
# }

# # Define the specific colors for molecule types
# custom_colors = {
#     "light blue": "#ADD8E6",
#     "navy": "#000080",
#     "green": "#008000",
#     "red": "#FF0000",
#     "orange": "#FFA500",
#     "grey": "#808080",
#     "dark brown": "#654321",
# }

# # Map the colors to molecule types
# unique_types = dataset["Type"].unique()
# type_colors = {mol_type: custom_colors[color] for mol_type, color in zip(unique_types, custom_colors)}

# # Generate scatter plots
# for prop_name, col_name in properties_to_plot.items():
#     plt.figure(figsize=(10, 6))
#     for mol_type in unique_types:
#         subset = dataset[dataset["Type"] == mol_type]
#         plt.scatter(
#             subset["Heavy Atoms"],
#             subset[col_name],
#             label=mol_type,
#             alpha=0.9,
#             s=50,
#             color=type_colors[mol_type],
#         )
#     plt.title(f"Scatter Plot of {prop_name} vs. Heavy Atoms")
#     plt.xlabel("Number of Heavy Atoms")
#     plt.ylabel(prop_name)
#     plt.legend(title="Molecule Type", loc="best")
#     plt.grid(alpha=0.3)

#     # Save the figure
#     file_name = f"C:/Users/eeo21/Desktop/GeneticAlgoPaperImages/{prop_name.replace(' ', '_').replace('/', '_')}_vs_Heavy_Atoms.png"
#     plt.savefig(file_name, dpi=300)
#     print(f"Saved scatter plot to {file_name}")

#     plt.show()
