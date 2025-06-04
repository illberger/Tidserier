# /Backtest/polymarket_fetcher.py

from typing import Optional

import httpx
import time
import numpy as np

# Lista med de fasta clob_token_ids för ett urval av BTC‐relaterade marknader
FIXED_CLOB_TOKEN_IDS = [
    "112540911653160777059655478391259433595972605218365763034134019729862917878641",  # BTC 2025 (slutpris $1M)
    "83894672511259544049673946661753374355328822374216474995072428966535091173758",  # US National BTC reserve 2025
    "93592949212798121127213117304912625505836768562433217537850469496310204567695",  # MicroStrategy sells BTC 2025
]

def fetch_5m_values_for_token(
    clob_token_id: str,
    start_time_s:   int,
    end_time_s:     int
) -> list[float]:
    """
    Hämtar 5-minuters pris-tidsserie (p-värden) för givet clob_token_id
    i intervallet [start_time_s, end_time_s] (Unix-sekunder).
    Returnerar en lista av floats (kan vara tom om inget finns).
    """
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market":   clob_token_id,
        "startTs":  start_time_s,
        "endTs":    end_time_s,
        "fidelity": 5  # 5 = 5-minuters-upplösning
    }
    try:
        r = httpx.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return []  # om något går fel, returnera tom lista

    history = js.get("history", [])
    if not isinstance(history, list):
        return []

    # Extrahera alla 'p'-värden
    return [point["p"] for point in history if "p" in point]


def compute_combined_sentiment(
    fetch_shift: int,
    max_days:    int,
    lookback_minutes: int = 1440
) -> Optional[float]:
    """
    Räknar ut ett aggregerat sentiment_score genom att hämta 5-minuters 'p'-värden
    för varje clob_token_id i FIXED_CLOB_TOKEN_IDS, över samma tidsintervall som WS-fetchen.

    - fetch_shift:      skift för vilken dag (0 = senaste dag, 1 = gårdagen osv).
    - max_days:         totala antalet dagar vi backtestar mot.
    - lookback_minutes: antalet minuter bakåt från slutet av fönstret (default=1440 ⇒ tvådagarsfönster).

    Returnerar medelvärdet av alla insamlade p-värden (flat list över alla marknader).
    Om INGA p-värden hämtas från någon marknad, returneras 1.0 som standard.
    """
    day_ms = 24 * 60 * 60 * 1000
    window_minutes = lookback_minutes * 2
    window_ms = window_minutes * 60 * 1000

    now_ms = int(time.time() * 1000)
    end_time_ms = now_ms - ((max_days - fetch_shift - 1) * day_ms)
    start_time_ms = end_time_ms - window_ms

    start_s = start_time_ms // 1000
    end_s = end_time_ms // 1000

    expected_count = int(window_minutes / 5)

    all_p_values = []
    for token_id in FIXED_CLOB_TOKEN_IDS:
        p_vals = fetch_5m_values_for_token(token_id, start_s, end_s)
        if p_vals:
            if len(p_vals) < expected_count:
                continue
            all_p_values.extend(p_vals)

    if not all_p_values:
        return None

    return float(np.mean(all_p_values))


"""
Testanrop.
"""
if __name__ == "__main__":
    sentiment = compute_combined_sentiment(fetch_shift=0, max_days=2)
    print("Sentiment_score =", sentiment)
