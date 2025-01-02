import pandas as pd
from math import log10

# Load dataset and lookup table
dataset_path = 'CyclicMoleculeBenchmarkingDataset_with_ViscosityColumns_AllTemperatures.csv'
lookup_table_path = 'VILookupTable.csv'

data = pd.read_csv(dataset_path)
data = data.dropna()
RefVals = pd.read_csv(lookup_table_path)

# Define GetKVI and GetDVI functions
def GetKVI_with_table(KVisc40, KVisc100, RefVals):
    if KVisc100 is None or KVisc40 is None:
        return 0

    if KVisc100 >= 2:
        # Retrieve L and H values
        RefVals['Diffs'] = abs(RefVals['KVI'] - KVisc100)
        RefVals_Sorted = RefVals.sort_values(by='Diffs')
        NearVals = RefVals_Sorted.head(2)

        KVIVals = sorted(NearVals['KVI'].tolist())
        LVals = sorted(NearVals['L'].tolist())
        HVals = sorted(NearVals['H'].tolist())

        # Perform Interpolation
        InterLVal = LVals[0] + (((KVisc100 - KVIVals[0]) * (LVals[1] - LVals[0])) / (KVIVals[1] - KVIVals[0]))
        InterHVal = HVals[0] + (((KVisc100 - KVIVals[0]) * (HVals[1] - HVals[0])) / (KVIVals[1] - KVIVals[0]))

        # Calculate KVI
        if KVisc40 >= InterHVal:
            return ((InterLVal - KVisc40) / (InterLVal - InterHVal)) * 100
        elif InterHVal > KVisc40:
            N = ((log10(InterHVal) - log10(KVisc40)) / log10(KVisc100))
            return (((10 ** N) - 1) / 0.00715) + 100
        else:
            return None

    return 0

def GetDVI_custom(DVisc40, DVisc100):
    try:
        S = (-log10((log10(DVisc40) + 1.2) / (log10(DVisc100) + 1.2))) / (log10(175 / 235))
        return 220 - (7 * (10 ** S))
    except:
        return 0

# Calculate KVI and DVI
data['KVI'] = data.apply(lambda row: GetKVI_with_table(row['Kinematic Viscosity_40C'], row['Kinematic_Viscosity_100C'], RefVals), axis=1)
data['DVI'] = data.apply(lambda row: GetDVI_custom(row['Dynamic_Viscosity_40C'], row['Dynamic_Viscosity_100C']), axis=1)

# Select relevant columns
selected_columns = [
    'SMILES',
    'Kinematic Viscosity_40C',
    'Kinematic_Viscosity_100C',
    'Dynamic_Viscosity_40C',
    'Dynamic_Viscosity_100C',
    'Type',
    'KVI',
    'DVI'
]
result_df = data[selected_columns]

# Save or display the resulting DataFrame
result_df.to_csv('VIComparisonOutput.csv', index=False)
print("KVI and DVI calculations complete. Results saved to 'output.csv'.")

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Programmatically extract all RDKit descriptors
def extract_all_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptor_funcs = {name: func for name, func in Descriptors.descList}
            features = {name: func(mol) for name, func in descriptor_funcs.items()}
            return features
        else:
            return {name: None for name, _ in Descriptors.descList}
    except Exception as e:
        print(f"Error processing SMILES: {smiles} | Error: {e}")
        return {name: None for name, _ in Descriptors.descList}

# Extract features for each SMILES in the dataset
features = data['SMILES'].apply(extract_all_features)
features_df = pd.DataFrame(features.tolist())

# Merge extracted features with the original dataset
data_with_features = pd.concat([data, features_df], axis=1)

# Drop columns with NaN or zero values, except KVI and DVI
columns_to_keep = ['KVI', 'DVI']
columns_to_drop = [
    col for col in data_with_features.columns 
    if col not in columns_to_keep and (
        data_with_features[col].isna().all() or 
        (data_with_features[col] == 0).all()
    )
]
data_cleaned = data_with_features.drop(columns=columns_to_drop)

# Save the cleaned dataset
output_cleaned_path = 'output_cleaned_with_features.csv'
data_cleaned.to_csv(output_cleaned_path, index=False)
print(f"Cleaned dataset saved to: {output_cleaned_path}")

# === Begin Analysis ===

color_map = {
    'Ester': 'red',
    'Ether': 'blue',
    'Aromatic Ester': 'green',
    'Aromatic': 'black',
    'Cyclic Paraffin': 'cyan',
    'Paraffin': 'grey',
    'Alkene': 'orange'  # Add more types and colors as needed
}

# Scatter plot of KVI vs DVI
# RatesvsShear, RvS3 = plt.subplots()
# for molecule_type in data['Type'].unique():
#     subset = data[data['Type'] == molecule_type]
#     plt.scatter(
#         subset['KVI'], 
#         subset['DVI'], 
#         label=molecule_type, 
#         color=color_map.get(molecule_type, 'gray'),  # Default to gray if type not in color_map
#     )

# plt.xlabel('KVI')
# plt.ylabel('DVI')
# plt.legend(loc='best')
# plt.grid(True)
# RvS3.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
# RvS3.grid(which = "minor", linestyle='--', linewidth=0.2)
# plt.minorticks_on()
# plt.xlim(-200, 400)
# plt.ylim(0, 200)
# plt.title('Scatter Plot of KVI vs DVI')
# plt.show()

# Correlation analysis
correlation = data_cleaned[['KVI', 'DVI']].corr()
print("Correlation Analysis:\n", correlation)

# Statistical comparison using t-test
t_stat, p_value = ttest_ind(data_cleaned['KVI'], data_cleaned['DVI'], nan_policy='omit')
print(f"Statistical Comparison (T-test): T-statistic={t_stat}, P-value={p_value}")

# Visualisation of differences
# data_cleaned['Absolute_Difference'] = abs(data_cleaned['KVI'] - data_cleaned['DVI'])
# RatesvsShear, RvS3 = plt.subplots()
# plt.hist(data_cleaned['Absolute_Difference'].dropna(), bins=200, alpha=1, color='b')
# plt.xlabel('Absolute Difference (KVI - DVI)')
# plt.grid(True)
# RvS3.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
# RvS3.grid(which = "minor", linestyle='--', linewidth=0.2)
# plt.ylabel('Frequency')
# plt.minorticks_on()
# plt.xlim(0, 500)
# plt.title('Distribution of Differences Between KVI and DVI')
# plt.show()

# Count of 0/negative values for each measure
zero_negative_kvi = (data_cleaned['KVI'] <= 0).sum()
zero_negative_dvi = (data_cleaned['DVI'] <= 0).sum()

print(f"Count of 0/negative values for KVI: {zero_negative_kvi}")
print(f"Count of 0/negative values for DVI: {zero_negative_dvi}")

# Save analysis results for review
analysis_output_path = 'output_analysis_cleaned.csv'
data_cleaned.to_csv(analysis_output_path, index=False)
print(f"Dataset with all analysis saved to: {analysis_output_path}")

import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt

# Load dataset
dataset_path = 'output_cleaned_with_features_training.csv'
data_cleaned = pd.read_csv(dataset_path)

# Drop rows with NaN values
data_cleaned = data_cleaned.dropna()

# Use all structural features as predictors (excluding target columns)
target_columns = ['KVI', 'DVI']
excluded_columns = target_columns + ['SMILES']  # Add more if necessary
structural_features = [col for col in data_cleaned.columns if col not in excluded_columns]

data_cleaned = data_cleaned[(data_cleaned['KVI'] > 0) & (data_cleaned['DVI'] > 0)]

# Extract predictors and target variables
X = data_cleaned[structural_features].fillna(0)  # Replace NaN with 0
y_kvi = np.log10(data_cleaned['KVI'])  # Apply log10 transformation
y_dvi = np.log10(data_cleaned['DVI'])  # Apply log10 transformation

# Bayesian optimization with XGBoost
def train_xgboost_with_bayesian_optimization(X, y, target_name):
    """
    Trains an XGBoost model with Bayesian optimization and returns the optimized model.
    """
    # Define search space for hyperparameters
    search_space = {
        'n_estimators': Integer(100, 5000),
        'max_depth': Integer(10, 100),
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
    }

    # 5-fold cross-validation setup
    bayes_search = BayesSearchCV(
        estimator=XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse'),
        search_spaces=search_space,
        n_iter=50,  # Number of parameter settings to try
        cv=2,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    bayes_search.fit(X_train, y_train)

    # Best model
    best_model = bayes_search.best_estimator_

    print(f"\n{target_name} Model Results:")
    print(f"Best Parameters: {bayes_search.best_params_}")

    return best_model

# Train models
kvi_model = train_xgboost_with_bayesian_optimization(X, y_kvi, 'KVI')
dvi_model = train_xgboost_with_bayesian_optimization(X, y_dvi, 'DVI')

# SHAP Analysis for KVI
explainer_kvi = shap.Explainer(kvi_model, X)
shap_values_kvi = explainer_kvi(X)

# Convert shap_values to a NumPy array if necessary
shap_values_array_kvi = shap_values_kvi.values  # Extract SHAP values as a NumPy array

# Ensure the number of features matches
assert shap_values_array_kvi.shape[1] == X.shape[1], "Mismatch between SHAP values and input data."

# Save SHAP Summary Plot for KVI (Bar Plot)
plt.figure()
shap.summary_plot(shap_values_kvi, X, plot_type="bar", max_display=10, show=False)  # Display top 10 features
plt.title("SHAP Summary Plot (Bar) - KVI")
plt.savefig("shap_summary_bar_kvi.png")
plt.tight_layout()
plt.close()

# Save SHAP Detailed Plot for KVI
plt.figure()
shap.summary_plot(shap_values_kvi, X, max_display=10, show=False)  # Display top 10 features
plt.title("SHAP Summary Plot - KVI")
plt.savefig("shap_summary_detailed_kvi.png")
plt.tight_layout()
plt.close()

# SHAP Analysis for DVI
explainer_dvi = shap.Explainer(dvi_model, X)
shap_values_dvi = explainer_dvi(X)

# Convert shap_values to a NumPy array if necessary
shap_values_array_dvi = shap_values_dvi.values  # Extract SHAP values as a NumPy array

# Ensure the number of features matches
assert shap_values_array_dvi.shape[1] == X.shape[1], "Mismatch between SHAP values and input data."

# Save SHAP Summary Plot for DVI (Bar Plot)
plt.figure()
shap.summary_plot(shap_values_dvi, X, plot_type="bar", max_display=10, show=False)  # Display top 10 features
plt.title("SHAP Summary Plot (Bar) - DVI")
plt.savefig("shap_summary_bar_dvi.png")
plt.tight_layout()
plt.close()

# Save SHAP Detailed Plot for DVI
plt.figure()
shap.summary_plot(shap_values_dvi, X, max_display=10, show=False)  # Display top 10 features
plt.title("SHAP Summary Plot - DVI")
plt.savefig("shap_summary_detailed_dvi.png")
plt.tight_layout()
plt.close()

print("SHAP plots have been saved.")
