# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the dataset
# file_path = "C:/Users/eeo21/Desktop/Datasets/TransformedDataset.csv"  # Update with the correct file path
# df = pd.read_csv(file_path)

# # Create a new folder for the datasets
# output_folder = "property_datasets"
# os.makedirs(output_folder, exist_ok=True)

# # Extract column names
# columns = df.columns

# # Identify SMILES column
# smiles_column = columns[0]  # Assuming first column is SMILES

# # Identify property columns
# property_columns = columns[1:]  # All other columns

# # Process each property separately
# for prop in property_columns:
#     # Create a new dataframe with SMILES and the selected property
#     df_prop = df[[smiles_column, prop]].dropna()

#     # Split into train (90%) and test (10%)
#     train_df, test_df = train_test_split(df_prop, test_size=0.1, random_state=42)

#     # Define file paths
#     train_path = os.path.join(output_folder, f"train_{prop}.csv")
#     test_path = os.path.join(output_folder, f"test_{prop}.csv")
    
#     # Save datasets
#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)

# print(f"Datasets saved in {output_folder}/")

