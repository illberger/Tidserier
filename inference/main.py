# main.py

import logging
import asyncio
import threading
import time

import joblib
from binance_client import BinanceWebSocketClient
from model_manager import ModelManager, load_model
logging.basicConfig(level=logging.INFO)

predictions = {}
true_records = []
"""
TODO::: TA in error pct in i update gradients.
"""

def subscribe_and_predict(ws_client, model_mng, symbol):
    symbol = symbol.upper()

    if not ws_client.is_streaming(symbol):
        ws_client.fetch_historical_5m_candles(symbol)
        ws_client.subscribe_to_klines([symbol], interval="5m")

    if symbol in predictions:
        print(f"{symbol} already has a prediction pending (ts={predictions[symbol][0]})")
        return

    def do_initial_predict():
        seq_scaled, scaler, seq_raw, last_open_ts = ws_client.get_latest_sequence(symbol, seq_len=288)
        if seq_scaled is None or not seq_scaled.any():
            print(f"{symbol} needs 288 candles before we can predict.")
            return

        pct, pred_close = model_mng.predict_close(symbol, seq_scaled, seq_raw)
        predictions[symbol] = (last_open_ts, pred_close)
        print(f"{symbol} → next_close={pred_close}, percent_change={pct}")

    t = threading.Timer(2.0, do_initial_predict)
    t.daemon = True
    t.start()


def monitor_predictions(ws_client, model_mng):
    while True:
        time.sleep(10)
        for symbol, (pred_open_ts, pred_close) in list(predictions.items()):
            res = ws_client.get_latest_sequence(symbol, seq_len=288)
            if not res or res[0] is None:
                continue
            _, _, x_raw, last_open_time = res
            if last_open_time > pred_open_ts:

                true_close = float(x_raw[-1, 3])  # 3 är index för close price
                error = true_close - pred_close
                true_records.append((symbol, pred_open_ts, true_close, error))
                del predictions[symbol]
                print(f"[True arrived] {symbol}@{pred_open_ts} → actual {true_close:.4f}, error {error:.4f}")
                subscribe_and_predict(ws_client, model_mng, symbol)


async def menu_loop(ws_client, model, model_mng):
    """
    A simple command-line menu to trigger predictions.
    """

    threading.Thread(
        target=monitor_predictions,
        args=(ws_client, model_mng),
        daemon=True
    ).start()

    while True:
        print("\n--- Menu ---")
        print("1. Stream via WS and start predict loop.")
        print("2. Save updated online-learning LSTM")
        print("Q. Quit")

        choice = input("Choose an option: ").strip().upper()

        if choice == "1":
            symbol_input = input("Enter symbol, e.g. 'BTCUSDT': ").strip().upper()
            subscribe_and_predict(ws_client, model_mng, symbol_input)

        elif choice == "2":
            print("Saving model...")
            try:
                model_mng.save_model()
            except IOError as e:
                print(f"Error: {e}")

        elif choice == "Q":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please select 1, 2, or Q.")


async def main():
    """
    Asynchronous due handling of websockets.
    :return:
    """

    # Load LSTM
    model = load_model()
    model_mng = ModelManager(model, 1e-6)
    scalers = joblib.load('files/symbol_scalers.joblib')

    # Instantiate WSS CLIENT
    ws_client = BinanceWebSocketClient(scalers)

    # MENU INPUT LOOP
    await menu_loop(ws_client, model, model_mng)

    # CLOSE WSS
    ws_client.stop()

    exit()

if __name__ == '__main__':
    asyncio.run(main())

