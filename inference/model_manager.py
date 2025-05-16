# model_manager.py

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
                X_batch = self.last_X_scaled[np.newaxis, ...]
                # Note that this SHOULD work since the model is already compiled before serialized
                self.model.train_on_batch(X_batch, y_true) # Note that this also RETURNS a loss
        self.last_X_scaled = X_scaled.copy()
        self.last_X_unscaled = X_unscaled.copy()

        inp = X_scaled.reshape((1, X_scaled.shape[0], N_FEATURES))
        y_pct_hat: float = self.model.predict(inp, verbose=0).reshape(-1)
        last_close = X_unscaled[-1, CLOSE_IDX]
        pred_close: float = last_close * (1 + y_pct_hat[0])
        return y_pct_hat, pred_close

    def save_model(self, path: str = "./files/onlinemodel.keras"):
        self.model.save(path)
        print(f"Model saved to {path}")




