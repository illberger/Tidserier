# db_fetcher.py
import pyodbc
import pandas as pd

class DBFetcher:
    """
    Fetches from a SQL database the scheme at which the data is inserted to in /FETCHING/
    """

    def __init__(self, server='localhost', database='BinanceDB'):
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )

    def fetch_candles(self):
        """
        Reads candlestick data from dbo.CandleSticks
        Returns a sorted DataFrame:
        Columns: [Symbol, Interval, OpenTime, OpenPrice, HighPrice, LowPrice,
                  ClosePrice, Volume, CloseTime]
        """
        query = """
        SELECT
            Symbol,
            Interval,
            OpenTime,
            OpenPrice,
            HighPrice,
            LowPrice,
            ClosePrice,
            CloseTime
        FROM dbo.CandleSticks
        ORDER BY Symbol, OpenTime ASC
        """
        with pyodbc.connect(self.conn_str) as conn:
            df = pd.read_sql(query, conn)

        return df
