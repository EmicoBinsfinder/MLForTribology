"""
Script to train a Decision Tree to predict molecule viscosity
"""

### Imports
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from IPython.display import Image
from tqdm import tqdm

# Progress tracking
tqdm.pandas()

### Load in dataset
# Dataset = pd.read_csv('Datasets/FinalDataset.csv')

### Creating Molecular Descriptors for Use with Decision Tree

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    calculator = MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calculator.CalcDescriptors(mol)
    return descriptors

# Dataset['Descriptors'] = Dataset['smiles'].progress_apply(smiles_to_descriptors)

# Remove Nan Rows which are rows where descriptor generation failed
# Dataset = Dataset.dropna(subset=['Descriptors'])

# # Create a DataFrame with descriptors and viscosity values only
# descriptors_df = pd.DataFrame(Dataset['Descriptors'].tolist())

# # Add viscosity column to descriptors DataFrame
# descriptors_df['Viscosity'] = Dataset['visco@40C[cP]']

# # Save Decision Tree Dataset
# descriptors_df.to_csv('Datasets/DecisionTreeDataset_313K.csv')

descriptors_df = pd.read_csv('Datasets/DecisionTreeDataset_313K.csv')

# Separate features and target variable
X = descriptors_df.drop(columns=['Viscosity'])
y = descriptors_df['Viscosity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree regressor
dt = DecisionTreeRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Train the final model using the best parameters
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optional: Print feature importances
feature_importances = best_model.feature_importances_
important_features = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)
print("Feature Importances:")
for feature, importance in important_features:
    print(f"{feature}: {importance}")

# Export the tree to DOT format
dot_data = export_graphviz(best_model, out_file=None,
                           feature_names=X.columns,
                           filled=True, rounded=True,
                           special_characters=True)

# Use pydotplus to convert the DOT file to a PNG image
graph = pydotplus.graph_from_dot_data(dot_data)
png_image = graph.create_png()

# Display the image
Image(png_image)
