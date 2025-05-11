import pyodbc
from binance.client import Client
from binance.enums import HistoricalKlinesType
import datetime
"""
Here using the community wrapper python-binance
'pip install python-binance'

Note that this code will download and insert approximately 1 million rows of data (depending on timespan configuration)
"""

def get_db_connection():
    """
    1. Database Connection
     # Adjust server/database names as needed
    :return:
    """

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=BinanceDB;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


def ms_to_datetime_str(ms):
    """
    4. Convert TransactTime to datetime for Binance
    # If your DB times are in ms, that's perfect for Binance
    :param ms:
    :return:
    """
    # If your TransactTime is in ms, convert to Python datetime
    dt = datetime.datetime.utcfromtimestamp(ms / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def fetch_candlesticks_from_binance(client, symbol, start_ms, end_ms):
    """
    5. Fetch Candlesticks from Binance
    Uses python-binance to get 5m candlesticks from 'start_ms' to 'end_ms'.
    """
    start_str = ms_to_datetime_str(start_ms)
    end_str = ms_to_datetime_str(end_ms)

    # binance expects symbol like "BTCUSDT"
    # Adjust if needed for "PONDBTC" etc
    # Also note we can pass 'limit' or rely on the time range

    klines = client.get_historical_klines(
        symbol=symbol.upper(),
        interval=Client.KLINE_INTERVAL_5MINUTE,
        start_str=start_str,
        end_str=end_str,
        klines_type=HistoricalKlinesType.SPOT
    )

    # Each kline is:
    # [
    #   1499040000000,      // Open time
    #   "0.01634790",       // Open
    #   "0.80000000",       // High
    #   "0.01575800",       // Low
    #   "0.01577100",       // Close
    #   "148976.11427815",  // Volume
    #   1499644799999,      // Close time
    #   "2434.19055334",    // Quote asset volume
    #   308,                // Number of trades
    #   "1756.87402397",    // Taker buy base asset volume
    #   "28.46694368",      // Taker buy quote asset volume
    #   "17928899.62484339" // Ignore
    # ]

    return klines


def insert_candlesticks_into_db(symbol, interval, klines):
    """
    6. Insert candlesticks into dbo.CandleSticks
    :param symbol:
    :param interval:
    :param klines:
    :return:
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert row by row
    for k in klines:
        open_time = k[0]
        open_price = float(k[1])
        high_price = float(k[2])
        low_price = float(k[3])
        close_price = float(k[4])
        volume = float(k[5])
        close_time = k[6]

        # Insert into CandleSticks table
        cursor.execute("""
            INSERT INTO dbo.CandleSticks
            (Symbol, Interval, OpenTime, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, CloseTime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, interval, open_time, open_price, high_price, low_price, close_price, volume, close_time
        ))
    conn.commit()
    cursor.close()
    conn.close()


def fetch_all_symbols(client):
    tickers = client.get_symbol_ticker()
    symbols = [ticker['symbol'] for ticker in tickers]
    return symbols


def main():

    client = Client(api_key="", api_secret="")

    symbols = fetch_all_symbols(client)

    base_start_unix = 1746396000000
    day_ms = 86400000
    import random

    random_days = random.randint(1, 5)
    end_time = base_start_unix + random_days * day_ms

    for idx, symbol in enumerate(symbols, 1):
        try:
            klines = fetch_candlesticks_from_binance(client, symbol, base_start_unix, end_time)
            if klines:
                insert_candlesticks_into_db(symbol, "5m", klines)
                print(f"[{idx}/{len(symbols)}] Inserted {len(klines)} klines for {symbol}")
            else:
                print(f"[{idx}/{len(symbols)}] No data for {symbol} in this period.")
        except Exception as e:
            print(f"[{idx}/{len(symbols)}] Error fetching klines for {symbol}: {e}")

    print("Completed inserting expanded dataset!")

if __name__ == "__main__":
    main()
