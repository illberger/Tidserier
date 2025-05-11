# main.py

import gc
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from db_fetcher import DBFetcher
from data_processor import DataProcessor
from tensorflow import keras
import joblib

from datetime import datetime

feature_names = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'TimeSin', 'TimeCos']


def plot_feature_violins(X):
    """
    Creates a violin plot for each feature by flattening over all samples and timesteps.
    """
    n_features = X.shape[2]
    data = [X[:, :, i].flatten() for i in range(n_features)]

    plt.figure(figsize=(12, 6))
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, n_features + 1), feature_names, rotation=45)
    plt.title("Violin Plots of Test Set Features")
    plt.ylabel("Scaled Feature Value")
    plt.tight_layout()
    plt.show()


def build_lstm_model(seq_len, num_features, label_width):
    inputs = keras.Input((seq_len, num_features))
    x = keras.layers.LSTM(32,
                          return_sequences=True,
                          dropout=0.2,
                          recurrent_dropout=0.2
                          )(inputs)
    x = keras.layers.LSTM(32,
                          dropout=0.2,
                          recurrent_dropout=0.2
                          )(x)
    outputs = keras.layers.Dense(label_width, activation='linear')(x)
    if label_width > 1:
        outputs = keras.layers.Reshape((label_width, 1))(outputs)
    return keras.Model(inputs, outputs)


def main():
    # 288 = 1 dag/25 H
    sequence_length = 288 # " N - SeqLen". Hur många candles modellen ska titta på i inputfönstret.
    train_ratio     = 0.7
    val_ratio       = 0.15
    # 1 feature = closePrice
    label_width = 1 # Antal tiddsteg att förutse;
    patience = 3 # Patience för early stopping (training)
    epochs = 50 # max epochs
    batch_size = 128 # Batcher att ta igenom varje träningsteg

    df = DBFetcher().fetch_candles()
    proc = DataProcessor(sequence_length, label_width)
    df_train, df_val, df_test = proc.split_by_time(df, train_ratio, val_ratio)

    x_train, y_train = proc.create_sequences(df_train)
    x_val,   y_val   = proc.create_sequences(df_val)
    x_test,  y_test  = proc.create_sequences(df_test)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   mode='min',
                                                   restore_best_weights=True)

    model = build_lstm_model(sequence_length, len(feature_names), label_width)
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    while True:
        _input = input("Model compiled, choose\n 1: Plot normalised data distributions\n2. Train NEW model\n3. Save model to file\n")
        match _input:
            case "1":
                n_features = x_test.shape[2]
                data = [x_test[:, :, i].flatten() for i in range(n_features)]

                plt.figure(figsize=(12, 6))
                plt.violinplot(data, showmeans=True)
                plt.xticks(range(1, n_features + 1), feature_names, rotation=45)
                plt.title("Violin Plots of Test Set Features")
                plt.ylabel("Scaled Feature Value")
                plt.tight_layout()
                plt.show()
            case "2":
                model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping]
                )
                loss, mae = model.evaluate(x_test, y_test)
                print(f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}")
            case "3":
                if model is None:
                    print("No current model to save. Please train a model first.")
                else:
                    filename = "../inference/files/model.keras"
                    joblib.dump(proc.scalers, '../inference/files/symbol_scalers.joblib')
                    model.save(filename)
                    print(f"Model saved to {filename}")
                    print(f"Scalers saved to symbol_scalers.joblib")


if __name__ == '__main__':
    main()


