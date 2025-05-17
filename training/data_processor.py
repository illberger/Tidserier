# data_processor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, sequence_length, label_width=1):
        self.sequence_length = sequence_length
        self.label_width = label_width
        self.scalers = {}

    def split_by_time(self, df, train_ratio, val_ratio):
        symbols = df['Symbol'].unique()
        n         = len(symbols)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        train_syms = symbols[:train_end]
        val_syms   = symbols[train_end:val_end]
        test_syms  = symbols[val_end:]

        def subdf(syms):
            return (df[df['Symbol'].isin(syms)]
                    .sort_values(['Symbol','OpenTime'])
                    .reset_index(drop=True))

        return subdf(train_syms), subdf(val_syms), subdf(test_syms)

    def create_sequences(self, df):
        all_X = []
        all_y = []
        features = ['OpenPrice','HighPrice','LowPrice','ClosePrice']

        """
        Olika symboler, olika normaliseringar.
        Sparar parametrarna för vardera MinMaxScaler i en dictionary, som sedan sparas till en fil.
        """
        for symbol, group in df.groupby('Symbol'):
            g = group.sort_values('OpenTime').reset_index(drop=True)
            close_raw = g['ClosePrice'].astype(float).values
            pct = np.empty_like(close_raw)
            pct[:-1] = close_raw[1:] / close_raw[:-1] -1
            pct[-1] = np.nan

            dt      = pd.to_datetime(g['OpenTime'], unit='ms', utc=True)
            minutes = dt.dt.hour * 60 + dt.dt.minute
            sin_t   = np.sin(2*np.pi * minutes / (24*60))
            cos_t   = np.cos(2*np.pi * minutes / (24*60))

            vals = g[features].astype(float).values  # (N,4)
            time_ft = np.vstack([sin_t, cos_t]).T  # (N,2)
            X_full = np.hstack([vals, time_ft])  # (N,6)

            scaler   = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_full)
            self.scalers[symbol] = scaler # save scaler

            N = len(X_scaled)
            for i in range(N - self.sequence_length - self.label_width + 1):
                seq_x = X_scaled[i: i + self.sequence_length]  # (L,6)
                # första target är pct vid index i+L-1
                start = i + self.sequence_length - 1
                seq_y = pct[start: start + self.label_width]  # (W,)

                if np.isnan(seq_y).any():
                    continue

                all_X.append(seq_x.astype(np.float32))
                all_y.append(seq_y.astype(np.float32))

        return np.array(all_X), np.array(all_y)

