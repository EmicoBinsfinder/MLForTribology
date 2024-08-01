import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import matplotlib.pyplot as plt
import ast

# Load the datasets
final_test_dataset = pd.read_csv('Datasets/FinalTestDataset.csv')
large_training_dataset = pd.read_csv('Datasets/LargeTrainingDataset.csv')
cnn_model_performance = pd.read_csv('PropertyPrediction/CNN/cnn_model_training_results.csv')

# Find the best performing model according to MSE
best_model_row = cnn_model_performance.loc[cnn_model_performance['Validation MSE'].idxmin()]

# Convert the row to a dictionary
best_params = best_model_row.to_dict()
model_params = ast.literal_eval(best_params['Model Params'])
train_params = ast.literal_eval(best_params['Train Params'])

# Merge model_params and train_params into a single dictionary
best_params = {**model_params, **train_params}
print(best_params)

# Extract best model parameters from the dictionary
optimizer = best_params['optimizer']
filters = int(best_params['filters'])
kernel_size = int(best_params['kernel_size'])
dense_units = int(best_params['dense_units'])
num_layers = int(best_params['num_layers'])
epochs = int(best_params['epochs'])
batch_size = int(best_params['batch_size'])

# Prepare the dataset
X = large_training_dataset['smiles']
y = large_training_dataset['visco@40C[cP]']

# Tokenize the smiles strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Reshape X to add an additional dimension for features
X = np.expand_dims(X, axis=-1)

# Prepare the test dataset
X_test = final_test_dataset['smiles']
y_test = final_test_dataset['visco@40C[cP]']
test_sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Reshape X_test to add an additional dimension for features
X_test = np.expand_dims(X_test, axis=-1)

# Function to create the CNN model
def create_cnn_model(optimizer, filters, kernel_size, num_layers, dropout_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1], 1)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

# Callbacks for model checkpointing and early stopping
callbacks = [
    ModelCheckpoint('cnn_checkpoint.model.keras', save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
]

# Determine the best dropout rate
dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
dropout_results = {}

for rate in dropout_rates:
    model = create_cnn_model(optimizer, filters, kernel_size, num_layers, rate)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    dropout_results[rate] = test_mse

best_dropout_rate = min(dropout_results, key=dropout_results.get)
print(f"Best dropout rate: {best_dropout_rate}")

# 5-fold cross-validation on different dataset sizes with the best dropout rate
dataset_sizes = [0.2, 0.4, 0.6, 0.8, 0.999]
cv_results = {size: [] for size in dataset_sizes}
cv_time = {size: [] for size in dataset_sizes}

for size in dataset_sizes:
    X_train, _, y_train, _ = train_test_split(X, y, train_size=size, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model = create_cnn_model(optimizer, filters, kernel_size, num_layers, best_dropout_rate)
        
        start_time = time.time()
        model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, validation_data=(X_fold_val, y_fold_val), callbacks=callbacks, verbose=0)
        end_time = time.time()
        
        test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_pred)
        cv_results[size].append(test_mse)
        cv_time[size].append(end_time - start_time)

# Plot the performance of 5-fold cross-validation at each dataset size
avg_cv_results = {size: np.mean(cv_results[size]) for size in cv_results}
std_cv_results = {size: np.std(cv_results[size]) for size in cv_results}

plt.figure(figsize=(10, 6))
plt.errorbar(dataset_sizes, [avg_cv_results[size] for size in dataset_sizes], 
             yerr=[std_cv_results[size] for size in dataset_sizes], fmt='-o', capsize=5)
plt.xlabel('Training Data Size')
plt.ylabel('Average Test MSE')
plt.title('5-Fold Cross-Validation Performance with Best CNN Model Using Test Set')
plt.grid(True)
plt.savefig('cnn_cv_performance_with_best_dropout.png')
plt.show()

# Save results to CSV
results_df = pd.DataFrame({
    'Training Data Size': dataset_sizes,
    'Average Test MSE': [avg_cv_results[size] for size in dataset_sizes],
    'Std Test MSE': [std_cv_results[size] for size in dataset_sizes],
    'Average Training Time (s)': [np.mean(cv_time[size]) for size in dataset_sizes]
})
results_df.to_csv('cnn_cv_results_with_best_dropout.csv', index=False)

# Print the results
print("Test MSE of the best CNN model with best dropout rate:", avg_cv_results)
