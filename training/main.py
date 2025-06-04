# /training/main.py

import matplotlib.pyplot as plt
from db_fetcher import DBFetcher
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_processor import preprocess_window
import pandas as pd
import numpy as np
"""
Polymarket Tags:

    Politics
    Crypto
    Sports
    Middle East
    Pop Culture
    Business
    Science
    All
"""




#<editor-fold desc="Globala konstanter">
feature_names = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'TimeSin', 'TimeCos']
sequence_length = 288
label_width = 1
loss = 'mse'
metrics = ['mae']
BATCH_SIZE = 512
LR = 1e-2
model = None
#</editor-fold>

"""
class BayTuner(kt.BayesianOptimization):
    
    def run_trial(self, trial, *args, **kwargs):
        bs = trial.hyperparameters.get('batch_size')
        kwargs['batch_size'] = bs
        return super().run_trial(trial,
                                 *args,
                                 **kwargs)
"""


def build_model(hp):
    """
    Defines model architecture and hp-span (lr and batch-size)
    Loss = huber
    Explicit initialization of weights

    TODO: Implementera varierbar tidsaxel
    inp = keras.Input(shape=(None, num_features))
    :param hp:
    :return:
    """
    inp = keras.Input((sequence_length, len(feature_names)))

    x = keras.layers.LSTM(
        32,
        return_sequences=True,
        #kernel_initializer='orthogonal',
        #recurrent_initializer='orthogonal'
        #kernel_regularizer=keras.regularizers.l2(1e-4),
        #recurrent_regularizer=keras.regularizers.l2(1e-4),
        #dropout=0.2,
        #recurrent_dropout=0.1
    )(inp)
    #x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LSTM(32)(x)
    #x = keras.layers.LayerNormalization()(x)
    #x = keras.layers.Dense(32, activation='relu',
    #                       kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    #x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(label_width, activation='linear')(x)

    cmodel = keras.Model(inp, out)
    cmodel.compile(optimizer=Adam(learning_rate=LR), loss=loss, metrics=metrics)

    if hp is None:
        lr = 1e-3
        cmodel.compile(
            optimizer=Adam(learning_rate=lr),
            loss='huber',
            metrics=['mae']
        )
        return cmodel
    # <editor-fold desc="Kompilering vid Bayesian Tuner från keras paketet (används ej, gammalt exempel)">
    else:

        lr = hp.Float('learning_rate',
                  min_value=1e-5,
                  max_value=1e-2,
                  sampling='log')
        hp.Choice('batch_size',
                   values=[32, 64, 128, 256, 512])

        cmodel.compile(
            optimizer=Adam(learning_rate=lr),
            loss='huber',
            metrics=['mae']
        )
        return cmodel
    # </editor-fold>


def pack_row(sym, ot, op, hi, lo, cl, vol):
    vec = tf.stack([tf.cast(ot, tf.float32), op, hi, lo, cl, vol], axis=0)
    vec.set_shape([6])          # <-- statisk form
    return vec

def main():
    #<editor-fold desc="Inita tuner (gammalt exempel)">




    print("TF:", tf.__version__, " NumPy:", np.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
    while True:
        _input = input("Choose\n 1: Plot normalised data distributions(OBSOLETE)\n2. Train NEW model\n3. Save model to file\n")
        match _input:
            case "1":
                print("Funktionen har tagits bort")
                #<editor-fold desc="Violin plottning">

                #</editor-fold>
            case "2":
                #<editor-fold desc="Bayesian Tuning Pipeline, Tidigare Exempel">

                #</editor-fold>

                # <editor-fold desc="Streaming from DB and model-fitting">
                model = build_model(hp=None) # Compiled model, 32 units LSTM

                fetcher = DBFetcher()

                total_rows = fetcher.row_count()
                total_sequences = total_rows - (sequence_length + label_width) + 1
                train_size = int(total_sequences * 0.6)
                val_size = int(total_sequences * 0.2)
                test_size = total_sequences - train_size - val_size

                full_ds = (
                    fetcher.get_dataset()
                    .map(pack_row, num_parallel_calls=tf.data.AUTOTUNE)
                    .window(sequence_length + label_width, shift=1, drop_remainder=True)
                    .flat_map(lambda w: w.batch(sequence_length + label_width))
                    .map(preprocess_window, num_parallel_calls=tf.data.AUTOTUNE)
                )

                gap = sequence_length + label_width - 1
                train_ds = (
                    full_ds
                    .take(train_size)
                    .shuffle(8641)  # Each SymbolPair is Grouped and sorted by the fetching code in groups of 8641 (doesn't mean that we have to shuffle by this number, but whatever)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE)
                )

                val_ds = (
                    full_ds
                    .skip(train_size + gap)
                    .take(val_size)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE)
                )

                test_ds = (
                    full_ds
                    .skip(train_size + gap + val_size + gap)
                    .take(test_size)
                    .batch(BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                )

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=100,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                                                           patience=3,
                                                             verbose=1,
                                                             restore_best_weights=True)]
                )

                mse_metric = tf.keras.metrics.MeanSquaredError()
                mae_metric = tf.keras.metrics.MeanAbsoluteError()
                for X_batch, y_batch in test_ds:
                    naive_preds = tf.expand_dims(X_batch[:, -1, 3], axis=1)
                    mse_metric.update_state(y_batch, naive_preds)
                    mae_metric.update_state(y_batch, naive_preds)

                baseline_mse_all = mse_metric.result().numpy()
                baseline_mae_all = mae_metric.result().numpy()
                print(f"Naiv baseline (hela test_ds) ‒ MSE: {baseline_mse_all:.5f}, MAE: {baseline_mae_all:.5f}")

                results = model.evaluate(test_ds)
                print("Test loss & metrics:", results)
                hist_df = pd.DataFrame(history.history)

                hist_df[['loss', 'val_loss']].plot(figsize=(8, 5))

                plt.title("Training vs Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend(["Train", "Val"])
                plt.show()
                # </editor-fold>
            case "3":
                if model is None:
                    print("No current model to save. Please train a model first.")
                else:
                    filename = "../inference/files/model.keras"
                    model.save(filename)
                    print(f"Model saved to {filename}")
                    print(f"Scalers saved to symbol_scalers.joblib")


if __name__ == '__main__':
    main()


