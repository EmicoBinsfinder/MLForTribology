import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import time
import matplotlib.pyplot as plt

# Load the datasets


final_test_dataset = pd.read_csv('Datasets/EBDatasetSMILES.csv')
large_training_dataset = pd.read_csv('Datasets/LargeTrainingDataset.csv')
model_performance = pd.read_csv('PropertyPrediction/MLP/model_performance.csv')

# Find the best performing model according to MSE
best_model_row = model_performance.loc[model_performance['avg_val_score'].idxmin()]

# Extract best model parameterswf 
optimizer = best_model_row['optimizer']
dense_units = best_model_row['dense_units']
num_layers = best_model_row['num_layers']
epochs = best_model_row['epochs']
batch_size = best_model_row['batch_size']

# Prepare the dataset
X = large_training_dataset['smiles']
y = large_training_dataset['visco@100C[cP]']

# Tokenize the smiles strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Prepare the test dataset
X_test = final_test_dataset['SMILES']
y_test = final_test_dataset['Experimental_100C_Viscosity']
test_sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Function to create the model with dropout
def create_model(optimizer, dense_units, num_layers, dropout_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(X.shape[1],)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

# Callback for learning rate scheduling and model checkpointing
callbacks = [
    LearningRateScheduler(scheduler),
    ModelCheckpoint('checkpoint.model.keras', save_best_only=True, monitor='val_loss')
]

# Define dropout rates to evaluate
dropout_rates = [0.0, 0.2]

# Store results for each dropout rate
dropout_results = {}

# Evaluate the model with different dropout rates
for dropout_rate in dropout_rates:
    best_model = create_model(optimizer, dense_units, num_layers, dropout_rate)
    best_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    dropout_results[dropout_rate] = test_mse

# Print test MSE for each dropout rate
print("Test MSE for each dropout rate:")
for rate, mse in dropout_results.items():
    print(f"Dropout rate {rate}: {mse}")

# 5-fold cross-validation on different dataset sizes for the best dropout rate
best_dropout_rate = min(dropout_results, key=dropout_results.get)
dataset_sizes = [0.999]
cv_results = {size: [] for size in dataset_sizes}
cv_time = {size: [] for size in dataset_sizes}

for size in dataset_sizes:
    X_train, _, y_train, _ = train_test_split(X, y, train_size=size, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model = create_model(optimizer, dense_units, num_layers, best_dropout_rate)
        
        start_time = time.time()
        model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, validation_data=(X_fold_val, y_fold_val), callbacks=callbacks, verbose=0)
        end_time = time.time()
        
        test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_pred)
        cv_results[size].append(test_mse)
        cv_time[size].append(end_time - start_time)

# Plot the performance of 5-fold cross-validation at each 20% increment of training data usage
avg_cv_results = {size: np.mean(cv_results[size]) for size in cv_results}
std_cv_results = {size: np.std(cv_results[size]) for size in cv_results}

plt.figure(figsize=(10, 6))
plt.errorbar(dataset_sizes, [avg_cv_results[size] for size in dataset_sizes], 
             yerr=[std_cv_results[size] for size in dataset_sizes], fmt='-o', capsize=5)
plt.xlabel('Training Data Size')
plt.ylabel('Average Test MSE')
plt.title('5-Fold Cross-Validation Performance with Best Dropout Rate Using Test Set')
plt.grid(True)
plt.savefig('cv_performance_with_dropout_Experimental_100C_MLP.png')
plt.show()

# Save results to CSV
results_df = pd.DataFrame({
    'Training Data Size': dataset_sizes,
    'Average Test MSE': [avg_cv_results[size] for size in dataset_sizes],
    'Std Test MSE': [std_cv_results[size] for size in dataset_sizes],
    'Average Training Time (s)': [np.mean(cv_time[size]) for size in dataset_sizes]
})
results_df.to_csv('cv_results_with_dropout_Experimental_100C_MLP.csv', index=False)

# Print the results
print("Best dropout rate:", best_dropout_rate)
print("Test MSE of the best model with best dropout rate:", dropout_results[best_dropout_rate])
