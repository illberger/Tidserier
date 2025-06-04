# /Backtest/main.py

import logging
import asyncio
import time

from matplotlib import pyplot as plt
import numpy as np
from polymarket_fetcher import compute_combined_sentiment
from binance_client import BinanceWebSocketClient
from model_manager import ModelManager, load_model
logging.basicConfig(level=logging.INFO)

predictions = {}
true_records = []
last_fetch_shift = -1
"""
NumPy v1.23.5
TensorFlow v2.10

Note that this code is highly unoptimized, since I'm fetching seq_length * 2 during each increment of
the fetch shift (where each previous window is always destroyed for "easy" memory management).
So each window is always fetched and evaluated twice, just to easily guarantee that every sequence of 288
always has atleast 288 + 1 for the statistics (need both the true of t+i (to plot and calculate error)
and the t-i series to feed into the LSTM for regression)

"""

def subscribe_and_predict(ws_client, model_mng, symbol, fetch_shift, slice_shift, max_days):
    global last_fetch_shift
    mean_bitcoin_sentiment = None  # Notera att endast Bitcoin sentiment kan hämtas enligt nuvarande konfig (om denna info är värdefull för något annat vet jag inte).
    symbol = symbol.upper()
    if fetch_shift > last_fetch_shift and fetch_shift < max_days:
        ws_client.closed_candles[symbol] = []  # Här töms objektets samling härifån
        ws_client.fetch_historical_5m_candles(symbol, fetch_shift, max_days, lookback_minutes=1440)
        """
        Funktionen nedan stödjer inte något symbolpar-argument, utan tar in tre "marknader" i samma upplösning som
        candlestick datan (den som LSTM-modellen är tränad på) relaterade till just Bitcoin. Det är oklart hur värdefull
        denna data är.
        """
        mean_bitcoin_sentiment = compute_combined_sentiment(fetch_shift, max_days)
        last_fetch_shift = fetch_shift
        while len(ws_client.closed_candles.get(symbol, [])) < 576:
            time.sleep(0.05)
    while True:
        seq_scaled, _, seq_raw, last_open_ts = ws_client.get_latest_sequence(
            symbol,
            slice_shift,
            seq_len=288
        )
        if seq_scaled is not None:
            break
        time.sleep(0.05)

    _, lstm_pred = model_mng.predict_close(symbol, seq_scaled, seq_raw, slice_shift, mean_bitcoin_sentiment, fetch_shift, max_days)
    _, naive_pred = model_mng.naive_forecast(symbol, seq_raw)
    return last_open_ts, lstm_pred, naive_pred

def monitor_predictions(ws_client, model_mng, symbol, max_days):
    """
    Main loop of the "Backtest".

    Utilizes two controlvariables "fetch_shift" and "slice_offset" (or slice_shift), which can simply be considered
    as outer loop controlvariable and inner loop controlvariable, respectively.

    Fetch_shift is simply used to count the shift of the time window of time t, or in simple terms: it is the index
    of the current time window in a 1d-array of max_days. The value of this index is used to subtract a value
    from a millisecond timestamp with respect to the current system time, and max_days (which can be considered as the
    length of this "1d-array").

    Slice_shift iterates 288 times for each iteration of fetch_shift. It is used to actually shift the time
    window t through each Sequences of Sequences. The value of slice_shift determines what actual candlestick (5 minute
    price bar) the model is predicting at each timestep. A.k.a. the time step.
    :param ws_client:
    :param model_mng:
    :param symbol:
    :param max_days:
    :return:
    """
    fetch_shift = 0
    slice_offset = 0
    last_ts, lstm, naive = subscribe_and_predict(ws_client, model_mng, symbol, fetch_shift, slice_offset, max_days)
    for test in range(max_days * 288):
        #time.sleep(0.1)
        seq = ws_client.get_latest_sequence(symbol, shift=slice_offset+1, seq_len=288)
        if not seq or seq[0] is None:
            continue
        _, _, seq_raw, current_ts = seq

        # Översätts till: if("Förekommer denna sekvens innan den tidigare sekvensen eller är det samma sekvens?") continue;
        #if current_ts <= last_ts and fetch_shift > 0:
        #    print(current_ts)
        #    print(last_ts)
        #    continue
        true_close = float(seq_raw[-1][3])
        true_records.append({
            'symbol':     symbol,
            'ts':         last_ts,
            'lstm_pred':  lstm,
            'naive_pred': naive,
            'true':       true_close,
            'error':      true_close - lstm
        })
        slice_offset += 1
        if slice_offset + 288 >= len(ws_client.closed_candles[symbol]):
            fetch_shift += 1  # or increment by 2 to optimize more, but need to redo other parts of code.
            slice_offset = 0
        #if test % 250 == 0:
            #  print(test)
        last_ts, lstm, naive = subscribe_and_predict(ws_client, model_mng, symbol, fetch_shift, slice_offset, max_days)


    idx     = np.arange(len(true_records))
    actuals = np.array([r['true']       for r in true_records])
    lstm_v  = np.array([r['lstm_pred']  for r in true_records])
    naive_v = np.array([r['naive_pred'] for r in true_records])
    lstm_err = np.array([r['true'] - r['lstm_pred'] for r in true_records])
    naive_err = np.array([r['true'] - r['naive_pred'] for r in true_records])
    def mae(e):
        return np.mean(np.abs(e))

    def rmse(e):
        return np.sqrt(np.mean(e ** 2))

    def mape(e):
        return np.abs(e) / actuals * 100

    mae_lstm = mae(lstm_err)
    mae_naive = mae(naive_err)
    rmse_lstm = rmse(lstm_err)
    rmse_naive = rmse(naive_err)
    mean_mape_lstm = np.mean(mape(lstm_err))
    mean_mape_naive = np.mean(mape(naive_err))

    lstm_mapes = mape(lstm_err)
    naive_mapes = mape(naive_err)
    low_thr_lstm = np.percentile(lstm_mapes, 33)
    high_thr_lstm = np.percentile(lstm_mapes, 66)
    low_thr_naive = np.percentile(naive_mapes, 33)
    high_thr_naive = np.percentile(naive_mapes, 66)

    def categorize(m, low, high):
        return np.where(m <= low, 'green',
                        np.where(m <= high, 'yellow', 'red'))

    colors_lstm = categorize(lstm_mapes, low_thr_lstm, high_thr_lstm)
    colors_naive = categorize(naive_mapes, low_thr_naive, high_thr_naive)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax1.plot(idx, actuals, color='gray', linewidth=0.5, label='True')
    ax1.scatter(idx, lstm_v, c=colors_lstm, s=2)
    text1 = (
        f"MAE  {mae_lstm:.2f}   RMSE  {rmse_lstm:.2f}\n"
        f"MAPE  {mean_mape_lstm:.2f}%"
    )
    ax1.text(0.02, 0.95, text1, transform=ax1.transAxes,
             fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    ax1.set_title(f"LSTM för {symbol}. {max_days} dagar.")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle=':', linewidth=0.5)

    # Baseline
    ax2.plot(idx, actuals, color='gray', linewidth=0.5, label='True')
    ax2.scatter(idx, naive_v, c=colors_naive, s=2)
    text2 = (
        f"MAE  {mae_naive:.2f}   RMSE  {rmse_naive:.2f}\n"
        f"MAPE  {mean_mape_naive:.2f}%"
    )
    ax2.text(0.02, 0.95, text2, transform=ax2.transAxes,
             fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    ax2.set_title("Baseline")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Price")
    ax2.grid(True, linestyle=':', linewidth=0.5)

    # Label
    global_text = (
        f"Över hela perioden:\n"
        f"LSTM MAE={mae_lstm:.2f}, MAPE={mean_mape_lstm:.2f} %   |   "
        f"Naive MAE={mae_naive:.2f}, MAPE={mean_mape_naive:.2f} %"
    )
    fig.text(0.5, 0.02, global_text,
             ha='center', va='bottom',
             fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()



async def menu_loop(ws_client, model, model_mng):
    """
    Backtest any symbol on binance market for any amount of time backwards.

    Will not finish if Binance is missing historical data for this symbol N amount of time backwards
    Loads 2 days of candlesticks into memory at all times, and slides a 288 (Sequence-length) window with a shift
    of 1 for each candlestick (first 288 candles are never predicted, since these are the initial sequence input)
    """

    monitor_predictions(ws_client, model_mng, "BTCUSDC", 365)  # Change args here <---



async def main():
    """
    Asynchronous due handling of websockets.
    :return:
    """

    # Load LSTM
    model = load_model()
    model_mng = ModelManager(model, 1e-6, 2.0, 0.2e-6, 0.1e-3)

    # Instantiate WSS CLIENT
    ws_client = BinanceWebSocketClient()

    # MENU INPUT LOOP
    await menu_loop(ws_client, model, model_mng)

    # CLOSE WSS
    ws_client.stop()

    exit()

if __name__ == '__main__':
    asyncio.run(main())

