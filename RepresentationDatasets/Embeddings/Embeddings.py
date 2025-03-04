import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
import tensorflow as tf
import shap
from keras.layers import Dense, Input
from keras.models import Model
from skopt import Optimizer
from skopt.space import Real, Integer

# Load the dataset
file_path = '/mnt/data/Concatenated_Molecular_Descriptors.csv'
df = pd.read_csv(file_path)

# Extract the molecular descriptors (assuming SMILES is in the 'SMILES' column)
descriptor_columns = [col for col in df.columns if col not in ['SMILES']]

# Extract SMILES strings and descriptors
smiles_data = df['SMILES']
descriptors = df[descriptor_columns].values

# Example target property for training (change as necessary)
target_property_columns = ['Density_40C', 'Viscosity_40C', 'Thermal_Conductivity_100C']
target_property = df[target_property_columns].values


# Autoencoder Model Definition
def build_autoencoder(input_dim, encoding_dim=100):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    # Encoder layer
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # Decoder layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    # Encoder model
    encoder = Model(input_layer, encoded)
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder, encoder

# Optimize the hyperparameters using Bayesian Optimization
def optimize_autoencoder(descriptors):
    input_dim = descriptors.shape[1]
    param_space = [
        Integer(16, 512, name='encoding_dim'),  # encoding dimension
        Real(0.0001, 0.1, name='learning_rate')  # learning rate
    ]
    
    def objective(params):
        encoding_dim, learning_rate = params
        autoencoder, encoder = build_autoencoder(input_dim, encoding_dim=encoding_dim)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
        
        # Fit the autoencoder
        autoencoder.fit(descriptors, descriptors, epochs=10, batch_size=32, verbose=0)
        
        # Get the reconstruction loss
        loss = autoencoder.evaluate(descriptors, descriptors, verbose=0)
        return loss
    
    # Run Bayesian Optimization
    optimizer = Optimizer(param_space)
    result = optimizer.maximize(objective, n_calls=10, random_state=42)
    
    best_params = result.x
    return best_params

# Optimize the autoencoder hyperparameters
best_params = optimize_autoencoder(descriptors)
print(f"Best Hyperparameters: Encoding Dim: {best_params[0]}, Learning Rate: {best_params[1]}")
