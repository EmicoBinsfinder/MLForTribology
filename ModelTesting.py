"""
Model testing
"""

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from xgboost import XGBRegressor
import joblib

descriptor_file_path = 'C:/Users/eeo21/Desktop/Datasets/Density_40C_Train_Descriptors.csv'
train_data = pd.read_csv(descriptor_file_path)
target_column = "Density_40C"  # Replace with the correct target column name
X = train_data.drop(columns=[target_column, 'SMILES'])
y = train_data[target_column]

SMILES = 'CCCCCCCCCCCCCCCCC=C(CCCC)c1ccccc1'
mol = Chem.MolFromSmiles(SMILES)

# Calculate descriptors
descriptor_dict = {}
for desc_name, desc_func in Descriptors.descList:
    try:
        descriptor_dict[desc_name] = desc_func(mol)
    except Exception as e:
        descriptor_dict[desc_name] = None  # Handle calculation errors

input_data = pd.DataFrame([descriptor_dict], columns=X.columns)

model = joblib.load('retrained_xgboost_model_Density_40C.joblib')
prediction = model.predict(input_data)
print(prediction[0])


# Define file paths
# training_results_path =  'C:/Users/eeo21/VSCodeProjects/MLForTribology/XGBoostModels/xgboost_Density_40C_training_results.csv'
# Load the CSV files
# training_results = pd.read_csv(training_results_path)

# Find the best hyperparameters based on the lowest MSE
# best_row = training_results.loc[training_results['Test MSE'].idxmin()][0]
# best_row = ast.literal_eval(best_row)
# print(type(best_row))

# best_params = {
#     'n_estimators': int(best_row['n_estimators']),
#     'learning_rate': best_row['learning_rate'],
#     'max_depth': int(best_row['max_depth']),
# }

# Prepare features and target

# Train the model using the best hyperparameters
# best_model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=42)
# best_model.fit(X, y)

# prediction = best_model.predict(input_data)
# print(prediction[0])


# output_model_path = "retrained_xgboost_model_Density_40C.joblib"
# joblib.dump(best_model, output_model_path)
