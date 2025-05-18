# model_manager.py
from collections import deque

import tensorflow as tf
import numpy as np
from collections import defaultdict, deque
#import keras # may be used later

CLOSE_IDX = 3 # [open, high, low, CLOSE. sin_t, cos_t]
N_FEATURES = 6 # len[]
SEQ_LEN = 288


def load_model(model_path: str="files/model.keras", online_path: str="files/onlinemodel.keras"):
    """
    Loads model into memory. Will try to load an "online model" (model whose weights have been updated during inference),
    if it does not exist; will load the default model. Note that program will crash if "model.keras" does not exist in
    project root.
    :param model_path:
    :param online_path:
    :return:
    """
    true_path:str = ""
    try:
        model = tf.keras.models.load_model(online_path)
        true_path = online_path
    except ValueError:
        model = tf.keras.models.load_model(model_path)
        true_path = model_path
    print("Model loaded from", true_path)
    return model


class ModelManager:
    """
    Handles online-training, predictions and saving of updated model to file.
    """
    def __init__(self, model, lr: float):
        self.last_X_scaled = {}
        self.model = model
        self.last_X_scaled = {}
        self.last_X_unscaled = {}
        self.replay = defaultdict(lambda: deque(maxlen=10))
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def predict_close(self, symbol: str,  X_scaled: np.ndarray, X_unscaled: np.ndarray):
        """
        UPDATE: 2025-05-18
        This class' fields are now dictionaries holding lists of the numpy arrays, so they MUST be keyed
        by symbol and then null checked as before - so the program can stream and auto-predict multiple symbols
        at the same time (previous version mixed input-vectors in the same field, corrupting results and batch loss).

        Note also that these dicts are not memory-managed like the candlesticks of 'BinanceClient' (yet!), so memory
        issues are to be expected after running for a couple of hours.

        # Textbody initial commit

        Predicts a percentage change [%] based on 288 candlesticks towards the next candlestick (5 minute closePrice).
        This %-change can be used to scale the last observed closePrice into an absolute closePrice-prediction. Models
        weights are updated if the last element of *this* instances' normalized/scaled input-sequence is not equal to
        the last element of param 'X_scaled'. After a prediction has been executed, this instances scaled-input-sequence
        (last_X_scaled) is given the values of the param 'X_scaled' (the latest input sequence).
        :param X_scaled:
        :param X_unscaled:
        :return:
        """

        symbol = symbol.upper()
        last_scaled = self.last_X_scaled.get(symbol)
        last_unscaled = self.last_X_unscaled.get(symbol)

        if last_scaled is not None and last_scaled.shape[0] == SEQ_LEN:
            if not np.array_equal(last_scaled[-1], X_scaled[-1]):
                prev_close = last_unscaled[-1, CLOSE_IDX]
                new_close = X_unscaled[-1, CLOSE_IDX]
                y_true = np.array([[new_close / prev_close - 1]], dtype=np.float32)
                x_batch = last_scaled[np.newaxis, ...]

                # Note that this SHOULD work since the model is already compiled before serialized
                batch_loss = self.online_update(symbol, x_batch, y_true)
                print(f"{symbol}, batch loss: {batch_loss}")
        self.last_X_scaled[symbol] = X_scaled.copy()
        self.last_X_unscaled[symbol] = X_unscaled.copy()

        inp = X_scaled.reshape((1, X_scaled.shape[0], N_FEATURES))
        y_pct_hat: float = self.model.predict(inp, verbose=0).reshape(-1)
        last_close = X_unscaled[-1, CLOSE_IDX]
        pred_close: float = last_close * (1 + y_pct_hat[0])
        return y_pct_hat, pred_close

    def online_update(self, symbol: str, X_batch, y_true):
        #
        """
        Update: 2025-05-18 Replay is now a dict keyed by the symbol in this method,
        updated in accord with the update of method 'predict_close' to support streaming and
        auto-predicting for multiple symbols.


        # Text-body initial commit

        y_true targets the close price of the features
        X_batch is 1 batch, seq_len is 288, n_Features is 6 (2 for time)
        :param model: keras model instance
        :param optimizer: keras optimizer instance
        :param X_batch: shape (1, seq_len, n_features);
        :param y_true: (1, 1);
        :return: Loss for this batch (for statistics or further computing)
        """
        if X_batch.shape[1] != SEQ_LEN:
            print(f"Warning: Recieved seqlen {X_batch.shape[1]} for {symbol} in online_update.")
            return 0.0

        replay = self.replay[symbol]
        replay.append((X_batch.squeeze(0), y_true.squeeze(0)))

        batch_x = np.stack([xb for xb, _ in replay], axis=0)
        batch_y = np.stack([yt for _, yt in replay], axis=0)

        loss_fn = tf.keras.losses.Huber(delta=0.002)
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = loss_fn(batch_y, y_pred)
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Intervall, fundera över detta.
        min_lr = self.lr  # 1e-6
        max_lr = 1e-2

        # Uppdatering av LR
        # skalär
        last_err = batch_y[-1] - y_pred[-1, 0]  # y_pred är (k,1)
        error = float(last_err)
        adj_lr = self.lr * (1 + abs(error)*2.0)
        adj_lr = max(min_lr, min(adj_lr, max_lr))
        self.optimizer.learning_rate.assign(adj_lr)

        # Uppdatering av gradienten
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()


    def save_model(self, path: str = "./files/onlinemodel.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")




