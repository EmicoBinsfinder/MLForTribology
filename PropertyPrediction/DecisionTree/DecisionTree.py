"""
Script to train a Decision Tree to predict molecule viscosity
"""

### Imports
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from IPython.display import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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
model = DecisionTreeRegressor(random_state=42)

# Cross-Validation Scores
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-np.mean(cv_scores)}")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure()
plt.title("Learning Curve for DecisionTreeRegressor")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.grid()

# Plot the mean train and test scores with standard deviation
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()
