You are an senior machine learning developer with experience in developing cheminformatics code.

Based on the provided dataset, can you please provide code that predicts the viscosity
of a molecule based on its SMILES representation, ensuring the code also does the following:

- Train Gaussian Mixture models to predict the viscosity of a molecule based on its SMILES string
- Performs a grid search of a range of different hyperparameters to optimise performance, using a custom grid search function
- Performs 5 fold cross validation for each model trained during the grid search
- Saves the performance of every model trained during the grid search to a csv file, saving after each model has been trained
- Measure the time it takes to train each model, saving this metric to the csv