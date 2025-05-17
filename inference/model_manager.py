# model_manager.py
from collections import deque

import tensorflow as tf
import numpy as np
#import keras # may be used later

CLOSE_IDX = 3 # [open, high, low, CLOSE. sin_t, cos_t]
N_FEATURES = 6 # len[]


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
        self.last_X_scaled = None
        self.model = model
        self.last_X_scaled = None
        self.last_X_unscaled = None
        self.replay = deque(maxlen=10)
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def predict_close(self, X_scaled: np.ndarray, X_unscaled: np.ndarray):
        """
        Predicts a percentage change [%] based on 288 candlesticks towards the next candlestick (5 minute closePrice).
        This %-change can be used to scale the last observed closePrice into an absolute closePrice-prediction. Models
        weights are updated if the last element of *this* instances' normalized/scaled input-sequence is not equal to
        the last element of param 'X_scaled'. After a prediction has been executed, this instances scaled-input-sequence
        (last_X_scaled) is given the values of the param 'X_scaled' (the latest input sequence).
        :param X_scaled:
        :param X_unscaled:
        :return:
        """
        if self.last_X_scaled is not None:
            if not np.array_equal(self.last_X_scaled[-1], X_scaled[-1]):
                prev_close = self.last_X_unscaled[-1, CLOSE_IDX]
                new_close = X_unscaled[-1, CLOSE_IDX]
                y_true = np.array([[new_close / prev_close - 1]], dtype=np.float32)
                x_batch = self.last_X_scaled[np.newaxis, ...]

                # Note that this SHOULD work since the model is already compiled before serialized
                batch_loss = self.online_update(x_batch, y_true)
                print(f"Batch loss: {batch_loss}")
        self.last_X_scaled = X_scaled.copy()
        self.last_X_unscaled = X_unscaled.copy()

        inp = X_scaled.reshape((1, X_scaled.shape[0], N_FEATURES))
        y_pct_hat: float = self.model.predict(inp, verbose=0).reshape(-1)
        last_close = X_unscaled[-1, CLOSE_IDX]
        pred_close: float = last_close * (1 + y_pct_hat[0])
        return y_pct_hat, pred_close

    def online_update(self, X_batch, y_true):
        #
        """
        TODO:
        y_true targets the close price of the features
        X_batch is 1 batch, seq_len is 288, n_Features is 6 (2 for time)
        :param model: keras model instance
        :param optimizer: keras optimizer instance
        :param X_batch: shape (1, seq_len, n_features);
        :param y_true: (1, 1);
        :return: Loss for this batch (for statistics or further computing)
        """

        self.replay.append((X_batch.squeeze(0), y_true.squeeze(0)))

        batch_x = np.stack([xb for xb, _ in self.replay], axis=0)
        batch_y = np.stack([yt for _, yt in self.replay], axis=0)

        loss_fn = tf.keras.losses.Huber()
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




