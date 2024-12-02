import os
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import shap
import matplotlib.pyplot as plt

# Define the properties with associated model and dataset names
properties = [
    ("Thermal_Conductivity_40C", "Thermal Conductivity at 40C"),
    ("Thermal_Conductivity_100C", "Thermal Conductivity at 100C"),
    ("Density_40C", "Density at 40C"),
    ("Density_100C", "Density at 100C"),
    ("Viscosity_40C", "Viscosity at 40C"),
    ("Viscosity_100C", "Viscosity at 100C"),
    ("Heat_Capacity_40C", "Heat Capacity at 40C"),
    ("Heat_Capacity_100C", "Heat Capacity at 100C")
]

# Folder to save individual PNG files
output_folder = "SHAP_Analysis_Results"
os.makedirs(output_folder, exist_ok=True)

# Dataframe to collect top 5 features for each property
summary_table = []

# Loop through each property
for model_name, display_name in properties:
    try:
        # Paths for model and descriptor data
        descriptor_file_path = f"C:/Users/eeo21/Desktop/Datasets/{model_name}_Train_Descriptors.csv"
        model_file = f"retrained_xgboost_model_{model_name}.joblib"
        
        # Load training data to align features
        train_data = pd.read_csv(descriptor_file_path)
        target_column = model_name
        X = train_data.drop(columns=[target_column, 'SMILES'])  # Drop target and SMILES
        y = train_data[target_column]
        
        # Load the model
        model = joblib.load(model_file)
        
        # Perform SHAP analysis on training data
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Get mean absolute SHAP values for top 5 features
        shap_importance = pd.DataFrame({
            'Feature': X.columns,
            'Mean SHAP Value': abs(shap_values.values).mean(axis=0)
        }).sort_values(by='Mean SHAP Value', ascending=False).head(5)
        
        # Append to summary table
        shap_importance['Property'] = display_name
        summary_table.append(shap_importance)
        
        # Filter X for the top 5 features
        top_features = shap_importance['Feature']
        X_top = X[top_features]
        shap_values_top = shap.Explanation(
            shap_values.values[:, top_features.index],
            base_values=shap_values.base_values,
            data=X_top,
            feature_names=top_features
        )
        
        # Save SHAP Summary Plot (Bar)
        plt.figure()
        shap.summary_plot(shap_values_top, X_top, plot_type="bar", show=False)
        plt.title(f"Top 5 SHAP Feature Importance (Bar) for {display_name}")
        plt.savefig(os.path.join(output_folder, f"{model_name}_bar.png"), bbox_inches='tight')
        plt.close()
        
        # Save SHAP Summary Plot (Beeswarm)
        plt.figure()
        shap.summary_plot(shap_values_top, X_top, show=False)
        plt.title(f"Top 5 SHAP Feature Importance (Beeswarm) for {display_name}")
        plt.savefig(os.path.join(output_folder, f"{model_name}_beeswarm.png"), bbox_inches='tight')
        plt.close()
    
    except Exception as e:
        print(f"Error processing {display_name}: {e}")

# Combine all top 5 features into a summary dataframe
summary_df = pd.concat(summary_table, ignore_index=True)

# Save and print the summary dataframe
summary_csv_path = os.path.join(output_folder, "Top_5_Features_Summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print("Top 5 Features Summary:")
print(summary_df)

