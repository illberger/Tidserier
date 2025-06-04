# /Backtest/model_manager.py
import itertools
from collections import deque

import tensorflow as tf
import numpy as np
from collections import defaultdict, deque


CLOSE_IDX = 3 # [open, high, low, close, vol, sin_t, cos_t]
N_FEATURES = 7 # len[]
SEQ_LEN = 288
TOTAL_UPDATES = 0

def load_model(model_path: str="files/model.keras", online_path: str="files/onlinemodel.keras"):
    """
    Loads model into memory. Will try to load an "online model" (model whose weights have been updated),
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
    except (ValueError, OSError):
        model = tf.keras.models.load_model(model_path)
        true_path = model_path
    print("Model loaded from", true_path)
    return model


class ModelManager:
    """
    Handles online-training, predictions and saving of updated model to file.
    """
    def __init__(self, model, lr: float, lam, min_lr, max_lr):
        self.model = model
        self.last_X_scaled = {}
        self.last_X_unscaled = {}
        self.replay = defaultdict(lambda: deque(maxlen=10))
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.max_replay = 10  # = 10 dagar
        self.lam = lam
        self.min_lr = min_lr
        self.max_lr = max_lr

    def naive_forecast(self, symbol: str, X_unscaled: np.ndarray):
        """
        Enkel naiv prognos: antar att nästa close-pris blir samma som senaste.
        Returnerar (percent_change, pred_close).
        """
        seq_unscaled = X_unscaled
        if seq_unscaled is None or seq_unscaled.shape[0] == 0:
            return 0.0, None

        last_close = seq_unscaled[-1, CLOSE_IDX]
        y_naive = 0.0
        pred_close_naive = last_close
        return y_naive, pred_close_naive

    def predict_close(self, symbol: str,  X_scaled: np.ndarray, X_unscaled: np.ndarray, slice_shift: int, sentiment_score: float | None, fetch_shift, max_days):
        """
        Vid anrop till "online_update" måste målvariabeln skalas om enligt "last_X"
        :param X_scaled:
        :param X_unscaled:
        :return:
        """
        if fetch_shift == (max_days):
            print(TOTAL_UPDATES)

        symbol = symbol.upper()
        last_scaled = self.last_X_scaled.get(symbol)
        last_unscaled = self.last_X_unscaled.get(symbol)

        if last_scaled is not None and last_scaled.shape[0] == SEQ_LEN:
            if not np.array_equal(last_scaled[-1], X_scaled[-1]):
                new_close_raw = X_unscaled[-1, CLOSE_IDX]

                all_closes_raw = last_unscaled[:, CLOSE_IDX]
                mu_last = np.mean(all_closes_raw)
                sigma_last = np.std(all_closes_raw)

                y_true_norm = (new_close_raw - mu_last) / sigma_last
                y_true = np.array([[y_true_norm]], dtype=np.float32)

                x_batch = last_scaled[np.newaxis, ...]
                #print(x_batch.shape)

                if slice_shift == 0:
                    print(f"sentiment_score={sentiment_score}")
                    if sentiment_score is not None:
                        _ = self.online_update(symbol, x_batch, y_true, sentiment_score)

        # Uppdatera last_X (senaste tidsteget (288,7))
        self.last_X_scaled[symbol] = X_scaled.copy()
        self.last_X_unscaled[symbol] = X_unscaled.copy()
        #print(X_scaled.shape)
        #print(len(self.last_X_scaled[symbol]))
        # Mata in X_scaled[t] in i modellen, och skala prediktionen enligt X_raw[t+1]
        inp = X_scaled.reshape((1, X_scaled.shape[0], N_FEATURES))
        y_vector = self.model.predict(inp, verbose=0).reshape(-1) # Normalized price prediction
        all_y_for_stat = X_unscaled[:, CLOSE_IDX]
        mean_close = np.mean(all_y_for_stat)
        std_close = np.std(all_y_for_stat)
        pred_close: float = y_vector[-1] * std_close + mean_close # Avnormaliserat price prediction
        last_close = all_y_for_stat[-1]
        y_pct = pred_close / last_close - 1.0 # Procentuell förändring (används inte)
        return y_pct, pred_close

    def online_update(self, symbol: str, X_batch, y_true, sentiment_score: float):
        """
        Justerar nätverk med dynamisk replay‐storlek beroende på sentiment_score,
        samt modulering av learning rate.
        """

        symbol = symbol.upper()
        if X_batch.shape[1] != SEQ_LEN:
            print(f"Warning: Received seqlen {X_batch.shape[1]} for {symbol} in online_update.")
            return 0.0

        if symbol not in self.replay:
            self.replay[symbol] = deque(maxlen=self.max_replay)

        self.replay[symbol].append((X_batch.squeeze(0), y_true.squeeze(0)))

        lam = 2.0
        factor = np.exp(lam * (0.5 - sentiment_score))

        N = len(self.replay[symbol])
        if N == 0:
            return 0.0

        max_factor = np.exp(lam * (0.5 - 0.0))
        min_factor = np.exp(lam * (0.5 - 1.0))
        prop = (factor - min_factor) / (max_factor - min_factor)
        prop = np.clip(prop, 0.05, 1.0)
        k = int(np.ceil(prop * N))
        replay_to_use = list(itertools.islice(self.replay[symbol], N - k, N))
        print("Replay längd: ", len(replay_to_use))
        batch_x = np.stack([xb for xb, _ in replay_to_use], axis=0)
        batch_y = np.stack([yt for _, yt in replay_to_use], axis=0)

        loss_fn = tf.keras.losses.Huber(delta=0.002)
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = loss_fn(batch_y, y_pred)
            """
            loss = exp(loss)
            Bara tre stycken "polymarket-marknader" (yes/no)-marknader används för sentiment, och medelvärdet
            av det positiva sentimentet för dessa marknader är generellt ganska lågt. För att få in den som en 
            "riktig" signal så boostas "sentiment-scoret" exponentiellt för att bestämma hur många replays som ska tas,
            samt hur stora gradienterna ska bli för denna batch.
            """
            scaled_loss = loss * factor
        grads = tape.gradient(scaled_loss, self.model.trainable_variables)



        grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
        print(f"[GRADS] sentiment={sentiment_score:.3f}, grad_norms={grad_norms}")

        min_lr = tf.constant(1e-7, dtype=tf.float32)
        max_lr = tf.constant(0.1e-3, dtype=tf.float32)
        sent = tf.cast(sentiment_score, tf.float32)

        factor_norm = (sent - 0.15) / (0.35 - 0.187)
        factor_clipped = tf.clip_by_value(factor_norm, 0.0, 1.0)

        log_min = tf.math.log(min_lr)
        log_max = tf.math.log(max_lr)

        adj_lr = tf.exp(log_min + factor_clipped * (log_max - log_min))
        adj_lr = tf.clip_by_value(adj_lr, min_lr, max_lr)

        self.optimizer.learning_rate.assign(adj_lr)

        print(f"[LR] sentiment={sentiment_score:.3f}, adj_lr={adj_lr:e}")

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        global TOTAL_UPDATES
        #if TOTAL_UPDATES < len(self.model.trainable_variables):
        #    print(len(grads), len(self.model.trainable_variables))
        #    print(grads[TOTAL_UPDATES].shape)
        TOTAL_UPDATES += 1
        return float(loss)


    def save_model(self, path: str = "./files/onlinemodel.keras"):
        """
        Saving a "online" model, etc
        :param path:
        :return:
        """
        self.model.save(path)
        print(f"Model saved to {path}")




