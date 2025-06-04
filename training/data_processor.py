# data_processor.py
import numpy as np
import tensorflow as tf
"""
Tensorflow v2.10.1
Numpy v1.23.5
"""


def preprocess_window(window):
    """
    "Förbehandlingsfönstret". Returnerar två st tensorflow-tensorer
    Data is already cleaned inside database in report example (removed SymbolPairs insufficient data using T-SQL)
    :param window:
    :return:
    """

    # Om man vill verifiera varje fönster/sekvens så kan man köra printsatsen nedan.
    #tf.print("preprocess_window körs på fönster med shape:", tf.shape(window))
    vals_all = window[:, 1:6]
    opentime = tf.cast(window[:, 0], tf.int64)
    minutes  = tf.cast((opentime // 60000) % (24*60), tf.float32)
    sin_t    = tf.sin(2*np.pi*minutes/(24*60))
    cos_t    = tf.cos(2*np.pi*minutes/(24*60))
    time_ft  = tf.stack([sin_t, cos_t], axis=1)

    X_full = tf.concat([vals_all, time_ft], axis=1)

    mean, var = tf.nn.moments(X_full, axes=[0])
    std       = tf.sqrt(var + 1e-6)
    X_scaled_full = (X_full - mean) / std

    unseen_close_price = X_scaled_full[-1, 3] # Z-Score Normalised Label Target

    # Raw closes
    unseen_close_price_raw = X_full[-1, 3]
    last_seen_close_price_raw = X_full[-2, 3]

    # Pct_change Label Target as %
    pct_change = ((unseen_close_price_raw - last_seen_close_price_raw) / last_seen_close_price_raw) * 100.0

    # Pct_change Label Target as factor
    pct_change_factor = ((unseen_close_price_raw - last_seen_close_price_raw) / last_seen_close_price_raw)

    label = unseen_close_price

    X_scaled = X_scaled_full[:-1, :]

    return X_scaled, tf.expand_dims(label, 0)



