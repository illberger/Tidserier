# main.py

import matplotlib.pyplot as plt
from db_fetcher import DBFetcher
from data_processor import DataProcessor
from tensorflow import keras
from bayes_opt import BayesianOptimization
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import joblib


feature_names = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'TimeSin', 'TimeCos']
sequence_length = 288
label_width = 1


class BayTuner(kt.BayesianOptimization):
    """
    Subclass for bayesian tuning
    """
    def run_trial(self, trial, *args, **kwargs):
        bs = trial.hyperparameters.get('batch_size')
        kwargs['batch_size'] = bs
        return super().run_trial(trial,
                                 *args,
                                 **kwargs)

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


def build_model(hp):
    """
    Defines model architecture and hp-span (lr and batch-size)
    Loss = huber
    Explicit initialization of weights

    TODO: Implement varierbar tidsaxel, för att bättre prediktera realtid.
    inp = keras.Input(shape=(None, num_features))
    :param hp:
    :return:
    """
    inp = keras.Input((sequence_length, len(feature_names)))

    x = keras.layers.LSTM(
        128,
        return_sequences=True,
        kernel_initializer='orthogonal',
        recurrent_initializer='orthogonal',
        kernel_regularizer=keras.regularizers.l2(1e-4),
        recurrent_regularizer=keras.regularizers.l2(1e-4),
        dropout=0.2,
        recurrent_dropout=0.1
    )(inp)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.LSTM(64, dropout=0.2)(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(label_width, activation='linear')(x)

    model = keras.Model(inp, out)

    if hp is None:
        lr = 1e-4
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='huber',
            metrics=['mae']
        )
        return model
    else:

        lr = hp.Float('learning_rate',
                  min_value=1e-5,
                  max_value=1e-2,
                  sampling='log')
        hp.Choice('batch_size',
                   values=[32, 64, 128, 256, 512]) # 64 är en OK batch_Size för detta

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='huber',
            metrics=['mae']
        )
        return model


def main():
    # 288 = 1 dag/25 H
    # sequence length är nu global var
    train_ratio     = 0.7
    val_ratio       = 0.15
    # 1 feature = closePrice
    # label width är nu global var
    patience = 3 # Patience för early stopping (training)
    epochs = 50 # max epochs
    model = None  # placeholder

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

    tuner = BayTuner(
        build_model,
        objective='val_loss',
        max_trials=10,
        num_initial_points=5,
        seed=42,
        directory='bay_tuner',
        project_name='lstm_may2025'
    )
    while True:
        _input = input("Choose\n 1: Plot normalised data distributions\n2. Train NEW model(hp search)\n3. Save model to file\n")
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
                """
                tuner.search(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=3,
                    callbacks=[early_stopping],
                    verbose=1
                )
                # 8896 steps för denna batch_size 32. Vad är nästa?
                best_hp = tuner.get_best_hyperparameters(1)[0]
                best_lr = best_hp.get('learning_rate')
                best_bs = best_hp.get('batch_size')
                print("Bästa lr:", best_lr)
                print("Bästa batch_size:", best_bs)

                model = tuner.get_best_models(1)[0]
                model.compile(optimizer=Adam(learning_rate=best_lr), loss='huber', metrics=['mae'])

                model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=best_bs,
                    callbacks=[early_stopping]
                )

                # Fryser första lagret innan export
                first_lstm = model.layers[1]
                first_lstm.trainable = False
                model.compile(optimizer=Adam(learning_rate=best_lr), loss='huber', metrics=['mae'])
                loss, mae = model.evaluate(x_test, y_test)
                print(f"Test: {loss:.4f}, Test MAE: {mae:.4f}")
                """
                model = build_model(hp=None)
                model.compile(optimizer=Adam(learning_rate=1e-4), loss='huber', metrics=['mae'])

                model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=64,
                    callbacks=[early_stopping]
                )
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


