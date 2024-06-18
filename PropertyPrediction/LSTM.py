import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Masking, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import shap

# Load Dataset
Dataset = pd.read_csv('Datasets/FinalDataset.csv')

# Tokenize SMILES strings
tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
tokenizer.fit_on_texts(Dataset['smiles'])
sequences = tokenizer.texts_to_sequences(Dataset['smiles'])
max_sequence_length = max(len(seq) for seq in sequences)

# Pad sequences to ensure uniform input length
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
y = Dataset['visco@40C[cP]'].values

# Define the hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'lstm_units': [32, 64, 128],
    'dense_units': [16, 32, 64],
    'num_lstm_layers': [1, 2],
    'bidirectional': [False, True],
    'epochs': [10, 20],
    'batch_size': [8, 16, 32]
}

# Function to create the model
def create_model(optimizer='adam', lstm_units=64, dense_units=32, num_lstm_layers=1, bidirectional=False):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length))
    model.add(Masking(mask_value=0.0))  # Mask padding values
    for _ in range(num_lstm_layers):
        if bidirectional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=(_ < num_lstm_layers - 1))))
        else:
            model.add(LSTM(lstm_units, return_sequences=(_ < num_lstm_layers - 1)))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap Keras model with scikit-learn's KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=0)

# Perform grid search with 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit grid search
grid_result = grid.fit(X, y, callbacks=[EarlyStopping(monitor='val_loss', patience=3), 
                                        ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)])

# Print best hyperparameters
print(f"Best hyperparameters: {grid_result.best_params_}")

# Train final model with best hyperparameters
best_params = grid_result.best_params_
best_model = create_model(best_params['optimizer'], best_params['lstm_units'], best_params['dense_units'],
                          best_params['num_lstm_layers'], best_params['bidirectional'])

# Save the best model
best_model.fit(X, y, epochs=best_params['epochs'], batch_size=best_params['batch_size'],
               callbacks=[ModelCheckpoint(filepath='best_model.h5', save_best_only=True)])

# Load the best model
best_model = load_model('best_model.h5')

# Make predictions with the best model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = best_model.predict(X_test)

# Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Best model MSE: {mse}')
print(f'Best model R2: {r2}')

# Plot the performance of each model tested during the grid search
results = pd.DataFrame(grid_result.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['mean_test_score'])
plt.xlabel('Model Index')
plt.ylabel('Mean Test Score (MSE)')
plt.title('Model Performance During Grid Search')
plt.show()

# Perform individual predictions based on input SMILES strings
def predict_smiles(smiles_string):
    sequence = tokenizer.texts_to_sequences([smiles_string])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = best_model.predict(padded_sequence)
    return prediction[0]

# Example prediction
example_smiles = "CCO"
print(f'Predicted viscosity for {example_smiles}: {predict_smiles(example_smiles)}')

# SHAP values for feature importance
explainer = shap.KernelExplainer(best_model.predict, X_train[:100])  # Using a small subset for explanation
shap_values = explainer.shap_values(X_test[:10])  # Again, a small subset for the example

# Plot SHAP values
shap.summary_plot(shap_values, X_test[:10], feature_names=tokenizer.index_word)

