# db_fetcher.py


import pyodbc
import tensorflow as tf


class DBFetcher:
    """
    Fetches from a SQL database the scheme at which the data is inserted to in /FETCHING/
    """

    def __init__(self, server='localhost', database='BinanceDB'):
        """
        Note that these parameters query a local SQL-server instance on my own PC,
        you would need to set up your own schema if you want to train model using this project (which is unlikely)
        :param server:
        :param database:
        """
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
        self.query = (
            "SELECT Symbol, OpenTime, "
            "OpenPrice, HighPrice, LowPrice, ClosePrice, Volume "
            "FROM dbo.CandleSticks "   # INCLUDE WHITESPACE HERE 
            "ORDER BY Symbol, OpenTime"
        )

    def row_count(self) -> int:
        """
        Räknar antalet rader i databasen
        :return:
        """
        """
        OBS! Dubbelkolla namnet på tabellen!
        """
        q = "SELECT COUNT(*) FROM dbo.CandleSticks"
        with pyodbc.connect(self.conn_str) as conn, conn.cursor() as cur:
            cur.execute(q)
            return cur.fetchone()[0]

    def _row_generator(self):
        """
        Intern generator som öppnar en kurs och yield:ar varje rad som en tuple.
        """
        conn = pyodbc.connect(self.conn_str)
        cur = conn.cursor()
        cur.execute(self.query)
        for row in cur:
            yield (
                row.Symbol,
                row.OpenTime,
                float(row.OpenPrice),
                float(row.HighPrice),
                float(row.LowPrice),
                float(row.ClosePrice),
                float(row.Volume),
            )
        conn.close()

    def get_dataset(self) -> tf.data.Dataset:
        """
        Returnerar ett tf.data.Dataset där varje element är:
           (symbol:str, openTime:int, open, high, low, close, volume)
        """

        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),   # Symbol
            tf.TensorSpec(shape=(), dtype=tf.int64),    # OpenTime
            tf.TensorSpec(shape=(), dtype=tf.float32),  # OpenPrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # HighPrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # LowPrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # ClosePrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # Volume
        )

        return tf.data.Dataset.from_generator(
            self._row_generator,
            output_signature=output_signature,
        )
