# main.py

import logging
import asyncio
import joblib
from binance_client import BinanceWebSocketClient
from model_manager import ModelManager, load_model
logging.basicConfig(level=logging.INFO)


async def menu_loop(ws_client, model, model_mng):
    """
    A simple command-line menu to trigger predictions.
    """
    while True:
        print("\n--- Menu ---")
        print("1) Subscribe to kline data for a symbol")
        print("2) Get the latest prediction for a symbol")
        print("3) Save updated online-learning LSTM")
        print("Q) Quit")

        choice = input("Choose an option: ").strip().upper()

        if choice == "1":
            symbols_input = input("Enter symbol(s), e.g. BTCUSDT ETHUSDT: ").strip().upper()
            symbols_list = symbols_input.split()
            for s in symbols_list:
                ws_client.fetch_historical_5m_candles(s)
            ws_client.subscribe_to_klines(symbols_list, interval="5m")

        elif choice == "2":
            symbol = input("Enter symbol for prediction (e.g. BTCUSDT): ").strip().upper()
            sequence_scaled, sc, sequence_raw = ws_client.get_latest_sequence(symbol)
            if sequence_scaled is not None and sc is not None:
                if sequence_scaled.any():
                    pct_change, next_close = model_mng.predict_close(sequence_scaled, sequence_raw)
                    print(f"Prediction for {symbol}: {next_close}")
                    print(f"% change for {symbol}: {pct_change}")
                else:
                    print(f"Not enough data for {symbol}. Have {len(ws_client.latest_candles.get(symbol, []))} candles stored.")
            else:
                print(f"No Klines or scaling data for {symbol}")
        elif choice == "3":
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

