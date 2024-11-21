'''
Analysing trainig results of Gradient Boosting for Viscosity Prediction
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'PropertyPrediction/GradientBoosting/GBResults.csv'
data = pd.read_csv(file_path)

# Extract hyperparameters into separate columns for analysis
data['learning_rate'] = data['Params'].apply(lambda x: eval(x)['learning_rate'])
data['max_depth'] = data['Params'].apply(lambda x: eval(x)['max_depth'])

# Pivot the data to create a matrix for heatmap
heatmap_data = data.pivot_table(values='Mean Test Score', index='learning_rate', columns='max_depth')

# Find the best hyperparameters
best_score = data['Mean Test Score'].min()
best_params = data.loc[data['Mean Test Score'].idxmin()]

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Mean Test Score'})
plt.title('Effect of Hyperparameters on Mean Test Score')
plt.xlabel('Max Depth')
plt.ylabel('Learning Rate')

# Mark the best hyperparameters
# plt.scatter(x=[best_params['max_depth']], y=[best_params['learning_rate']], color='red', label='Best Params')
# plt.legend()

plt.show()
