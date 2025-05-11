# binance_client.py
"""
Requires running 'pip install binance-connector'
"""
import logging
from collections import defaultdict
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
import json
import datetime as dt
import time
from math import sin, cos, pi
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class BinanceWebSocketClient:
    """
    A wrapper around the Binance SpotWebsocketStreamClient to subscribe to
    candlestick (kline) data for multiple symbols. Maintains a dictionary
    of the latest candles in memory, keyed by symbol.
    """
    #PARTIAL_MOD = 10

    def __init__(self, scalers: dict):
        self.current_history_symbol = None
        self.logger = logging.getLogger(__name__)
        self.closed_candles = {}
        self.partial_ctr = defaultdict(int)
        self.scalers = scalers # joblib scaler for each grouping in training
        self.LIMIT_CANDLES_STORAGE = 1000
        self.hist_client = SpotWebsocketAPIClient(
            on_message=self.hist_msg_handler,
            on_close=self.has_closed()
        )

        self.client = SpotWebsocketStreamClient(
            on_message=self._on_message,
            is_combined=True,
        )

    def _on_message(self, _, raw: str):
        """
        Lyssnare som bifogar stängda candlesticks som skickas från binances servrar varje 5-minuts-klockslag. WS-meddelanden
        kommer i JSON-format ungefär varje sekund. Se text-block nedan i denna metod gällande detta.
        :param _:
        :param raw:
        :return:
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.error("bad json")
            return

        data = msg.get("data", {})
        if data.get("e") != "kline":
            return

        k = data["k"]
        symbol = k["s"]
        closed = k["x"]

        if closed:
            self.closed_candles[symbol].append(k)
            self.closed_candles[symbol] = self.closed_candles[symbol][-self.LIMIT_CANDLES_STORAGE:]
            self.partial_ctr[symbol] = 0
            return
        """
        # Om man vill ta in mer information, exempelvis varje i:e candle (denna WSS-endpoint skickar tickers var 1:e sekund),
        # kan man avkommentera följande kod. Det betyder dock att Sin-, Cos-signalerna inte kommer passas in på det modellen
        # sett under träning. Detta hade kanske kunnat lösas med att överskugga candle_to_row där tid-nämnaren är den nya
        # tidrutan för "partial candles". Alternativt så hade man fått träna modellen på kanske 1-minute-candlesticks 
        # istället. 
        # Partial k-lines every 10th msg
        self.partial_ctr[symbol] += 1
        if self.partial_ctr[symbol] % self.PARTIAL_MOD != 0:
            return

        semi = k.copy() # "k" är hela K-linan med OHLC etc
        semi["t"] = data["E"]  # Replace the openTime before appending. # Realtiden finns innan "k"
        semi["x"] = True  # Ej essential metadata (används inte i träningen).
        self.closed_candles[symbol].append(semi)
        self.closed_candles[symbol] = self.closed_candles[symbol][-200:]
        """


    def fetch_historical_5m_candles(self, symbol: str, lookback_minutes=1440):
        """
        Ask the historical websocket client to fetch 5m candles for the specified lookback period.
        The returned JSON will be passed to hist_msg_handler.
        """
        symbol = symbol.upper()
        if symbol not in self.closed_candles:
            self.closed_candles[symbol] = []
        # Store the symbol as the current symbol for historical data
        self.current_history_symbol = symbol

        client = self.hist_client
        end_time = int(time.time() * 1000)  # current timestamp in ms
        start_time = end_time - (lookback_minutes * 60 * 1000)

        # Request 36 candles (180 minutes / 5 minutes = 36)
        client.klines(
            symbol=symbol,
            interval="5m",
            startTime=start_time,
            endTime=end_time,
            limit=int(lookback_minutes/5)
        )

    def hist_msg_handler(self, _, message: str):
        """
        Lyssnare för historiska candlesticks.
        :param _:
        :param message:
        :return:
        """
        print(message)
        try:
            message = json.loads(message)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode historical message: {message}")
            return

        # Check if this is a historical response
        if "result" not in message:
            return

        results = message["result"]
        for arr in results:
            # Each array arr has the format:
            # [open_time, open, high, low, close, volume, close_time, quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
            kline_dict = {
                't': arr[0],  # open time
                'o': arr[1],  # open price
                'h': arr[2],  # high price
                'l': arr[3],  # low price
                'c': arr[4],  # close price
                # 'T': arr[5],  # close time # Not included in training features
                'x': True,  # historical candles are closed
                's': message.get("symbol", self.current_history_symbol or "UNKNOWN"),
                'i': "5m"  # interval
            }

            # Override if symbol still reads "UNKNOWN"
            if kline_dict['s'] == "UNKNOWN":
                kline_dict['s'] = self.current_history_symbol or "UNKNOWN"

            symbol_key = kline_dict['s']
            if symbol_key not in self.closed_candles:
                self.closed_candles[symbol_key] = []
            self.closed_candles[symbol_key].append(kline_dict)

        # After processing, sort and trim the list:
        for sym in self.closed_candles:
            self.closed_candles[sym].sort(key=lambda x: x['t'])
            self.closed_candles[sym] = self.closed_candles[sym][-self.LIMIT_CANDLES_STORAGE:]

        self.logger.info(
            f"Historical data loaded: {{ {', '.join(f'{sym}: {len(self.closed_candles[sym])}' for sym in self.closed_candles)} }}")

    def has_closed(self):
        """
        Är tänkt att vara en callback för när en viss WSS-klient har stängts (ej lyssnare). Denna är ej klar.
        :return:
        """
        print("A WSS client has closed.")

    def subscribe_to_klines(self, symbols, interval="5m"):
        """
        Använder realtids-WSS-klienten för att streama candlesticks för en viss symbol.
        :param symbols:
        :param interval:
        :return:
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols:
            self.client.kline(symbol.lower(), interval=interval)
            self.logger.info(f"Subscribed to {symbol} @ {interval} klines.")

    def get_latest_sequence(self, symbol: str, seq_len=288):
        """
        Gets a sequence of 288 candlesticks (24h) from t-24 to t (now), t-window updates every 5 minutes,
        since it is a 5-minute candlestick.
        :param symbol:
        :param seq_len:
        :return:
        """
        rows = self.closed_candles.get(symbol.upper(), [])[-seq_len:]
        if len(rows) < seq_len:
            return None, None
        x = np.array([self.candle_to_row(k) for k in rows], dtype=np.float32)
        scaler: MinMaxScaler = self.scalers.get(symbol.upper())
        if scaler is None:
            return None, None  # symbol som inte fanns vid träning
        x_scaled = scaler.transform(x)  # shape (seq_len, 6)

        return x_scaled, scaler, x



    def candle_to_row(self, k):
        """
            Takes a whole candlestick and returns the columns expected by LSTM
            Also makes two new columns based on the parameter "OpenTime".
            {[N][6]} == [i][O, H, L, C, SIN(TimeOfDay), COS(TimeOfDay)]
            """
        # 5-minuters-stämplar i ms → UTC-minute-of-day
        minutes = (dt.datetime.utcfromtimestamp(k['t'] / 1000)
                   .hour * 60 +
                   dt.datetime.utcfromtimestamp(k['t'] / 1000).minute)
        sin_t = sin(2 * pi * minutes / (24 * 60))
        cos_t = cos(2 * pi * minutes / (24 * 60))
        return [float(k['o']), float(k['h']), float(k['l']),
                float(k['c']), sin_t, cos_t]

    def stop(self):
        """
        Stops all active streams.
        """
        self.logger.info("Stopping all websocket streams.")
        self.client.stop()
