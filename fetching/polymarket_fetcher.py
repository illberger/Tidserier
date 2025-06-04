# polymarket_fetcher.py.py
import time

import httpx
import os
import datetime as dt
import pandas as pd
import json
import pyodbc
from typing import List, Dict

"""
[{'id': '516861', 'question': 'Will Bitcoin reach $1,000,000 by December 31, 2025?', 'conditionId': '0xd8b9ff369452daebce1ac8cb6a29d6817903e85168356c72812317f38e317613', 'slug': 'will-bitcoin-reach-1000000-by-december-31-2025', 'resolutionSource': '', 'endDate': '2025-12-31T12:00:00Z', 'liquidity': '439628.22252', 'startDate': '2024-12-30T22:00:50.534999Z', 'image': 'https://polymarket-upload.s3.us-east-2.amazonaws.com/bitcoin+party111.png', 'icon': 'https://polymarket-upload.s3.us-east-2.amazonaws.com/bitcoin+party111.png', 'description': 'This market will immediately resolve to "Yes" if any Binance 1 minute candle for Bitcoin (BTCUSDT) between December 30, 2024, 20:00 and December 31, 2025, 23:59 in the ET timezone has a final "High" price of $1,000,000 or higher. Otherwise, this market will resolve to "No."\n\nThe resolution source for this market is Binance, specifically the BTCUSDT "High" prices available at https://www.binance.com/en/trade/BTC_USDT, with the chart settings on "1m" for one-minute candles selected on the top bar.\n\nPlease note that the outcome of this market depends solely on the price data from the Binance BTCUSDT trading pair. Prices from other exchanges, different trading pairs, or spot markets will not be considered for the resolution of this market.', 'outcomes': '["Yes", "No"]', 'outcomePrices': '["0.0295", "0.9705"]', 'volume': '881342.608817', 'active': True, 'closed': False, 'marketMakerAddress': '', 'createdAt': '2024-12-30T20:41:14.977618Z', 'updatedAt': '2025-06-01T17:10:25.118679Z', 'new': False, 'featured': False, 'submitted_by': '0x91430CaD2d3975766499717fA0D66A78D814E5c5', 'archived': False, 'resolvedBy': '0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74', 'restricted': True, 'groupItemTitle': '$1,000,000', 'groupItemThreshold': '0', 'questionID': '0x0422835a21786de482a3b7efd8cdb222e211ce392e8b2a975d9577d25ef5dc86', 'enableOrderBook': True, 'orderPriceMinTickSize': 0.001, 'orderMinSize': 5, 'volumeNum': 881342.608817, 'liquidityNum': 439628.22252, 'endDateIso': '2025-12-31', 'startDateIso': '2024-12-30', 'hasReviewedDates': True, 'volume24hr': 670.6591009999999, 'volume1wk': 28600.818684000027, 'volume1mo': 218370.4530190006, 'volume1yr': 881286.7994879994, 'clobTokenIds': '["112540911653160777059655478391259433595972605218365763034134019729862917878641", "72957845969259179114974336105989648762775384471357386872640167050913336248574"]', 'umaBond': '500', 'umaReward': '5', 'volume24hrClob': 670.6591009999999, 'volume1wkClob': 28600.818684000027, 'volume1moClob': 218370.4530190006, 'volume1yrClob': 881286.7994879994, 'volumeClob': 881342.608817, 'liquidityClob': 439628.22252, 'acceptingOrders': True, 'negRisk': False, 'events': [{'id': '16096', 'ticker': 'what-price-will-bitcoin-hit-in-2025', 'slug': 'what-price-will-bitcoin-hit-in-2025', 'title': 'What price will Bitcoin hit in 2025?', 'description': 'This is a market group over whit prices Bitcoin will hit in 2025.', 'resolutionSource': '', 'startDate': '2024-12-30T22:06:34.105524Z', 'creationDate': '2024-12-30T22:06:34.105521Z', 'endDate': '2025-12-31T12:00:00Z', 'image': 'https://polymarket-upload.s3.us-east-2.amazonaws.com/BTC+fullsize.png', 'icon': 'https://polymarket-upload.s3.us-east-2.amazonaws.com/BTC+fullsize.png', 'active': True, 'closed': False, 'archived': False, 'new': False, 'featured': False, 'restricted': True, 'liquidity': 1316943.63737, 'volume': 11760100.912182, 'openInterest': 0, 'createdAt': '2024-12-29T18:10:09.435812Z', 'updatedAt': '2025-06-01T17:08:42.917494Z', 'competitive': 0.9880446596186148, 'volume24hr': 143050.94929699998, 'volume1wk': 1016128.589026, 'volume1mo': 3731471.489491, 'volume1yr': 11240964.484263996, 'enableOrderBook': True, 'liquidityClob': 1316943.63737, 'negRisk': False, 'commentCount': 145, 'cyom': False, 'showAllOutcomes': True, 'showMarketImages': False, 'enableNegRisk': False, 'automaticallyActive': True, 'gmpChartMode': 'default', 'negRiskAugmented': False, 'pendingDeployment': False, 'deploying': False}], 'ready': False, 'funded': False, 'acceptingOrdersTimestamp': '2024-12-30T21:59:40Z', 'cyom': False, 'competitive': 0.8187525445293923, 'pagerDutyNotificationEnabled': False, 'approved': True, 'clobRewards': [{'id': '12375', 'conditionId': '0xd8b9ff369452daebce1ac8cb6a29d6817903e85168356c72812317f38e317613', 'assetAddress': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174', 'rewardsAmount': 0, 'rewardsDailyRate': 2, 'startDate': '2024-12-29', 'endDate': '2500-12-31'}], 'rewardsMinSize': 50, 'rewardsMaxSpread': 3.5, 'spread': 0.001, 'oneDayPriceChange': -0.001, 'oneWeekPriceChange': -0.005, 'oneMonthPriceChange': 0.002, 'lastTradePrice': 0.029, 'bestBid': 0.029, 'bestAsk': 0.03, 'automaticallyActive': True, 'clearBookOnStart': True, 'seriesColor': '', 'showGmpSeries': False, 'showGmpOutcome': False, 'manualActivation': False, 'negRiskOther': False, 'umaResolutionStatuses': '[]', 'pendingDeployment': False, 'deploying': False, 'rfqEnabled': False}]
"""


import httpx
import numpy as np

"""
cond_id_BTC1 = "0xd8b9ff369452daebce1ac8cb6a29d6817903e85168356c72812317f38e317613"  # Will Bitcoin reach $1,000,000 by Dec 31, 2025?
    cond_id_BTC2 = "0x80026f98f9de40aea8dba02798c4f0067942bba401fa3715209ee7c27482640b"  # US National BTC reserve in 2025?
    cond_id_BTC3 = "0x19ee98e348c0ccb341d1b9566fa14521566e9b2ea7aed34dc407a0ec56be36a2"  # MicroStrategy sells any BTC in 2025?
"""



"""
def fetch_markets_by_tag(tag_id: int) -> list[dict]:
    
    #Hämtar alla Markets från Gamma‐Markets‐API:t som har tag_id=tag_id.
    #Parametern related_tags=true säkerställer att vi får med taggarna för varje market
    #(så att vi kan se vilka tags som är associerade).
    #Returnerar en lista av dicts, där varje dict innehåller minst "id" och "slug".
    
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        #"tag_id": str(tag_id),
        #"related_tags": "true",
        "limit": "1000"
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            markets = response.json()
            print(response.json)
    except Exception as e:
        print(f"[fetch_markets_by_tag] Fel vid GET /markets: {e}")
        return []

    result = []
    for m in markets:
        market_id = m.get("id")
        slug      = m.get("slug")
        if market_id is None or slug is None:
            continue
        result.append({"id": market_id, "slug": slug, "tags": m.get("tags", [])})
    return result
"""

def get_clob_tokens_for_condition(condition_id: str) -> list[str] | None:
    """
    Hämtar market‐objektet för given condition_id och returnerar
    en lista av clobTokenIds (som strängar). Returnerar None om inget hittas.
    """
    url    = "https://gamma-api.polymarket.com/markets"
    params = {"condition_ids": condition_id}

    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        markets = resp.json()
    except Exception as e:
        print(f"[get_clob_tokens_for_condition] Fel vid GET /markets: {e}")
        return None

    if not markets:
        print(f"Inget market hittades för condition_id={condition_id}")
        return None

    m = markets[0]
    raw = m.get("clobTokenIds")
    if not raw:
        print(f"Market‐objekt saknar 'clobTokenIds' för condition_id={condition_id}")
        return None

    try:
        token_list = json.loads(raw)
    except Exception as e:
        print(f"Fel när vi försöker json.loads('clobTokenIds'): {e}")
        return None

    return token_list  # lista av token‐ID som strängar


def fetch_5m_values_for_token(
    clob_token_id: str,
    start_time_s:   int,
    end_time_s:     int
) -> list[float]:
    """
    Hämtar 5-minuters pris‐tidsserie för det givna clob_token_id
    i intervallet [start_time_s, end_time_s] (i sekunder sedan epoch).
    Returnerar en lista av alla 'p'-värden (float), eller en tom lista om inget finns.
    """
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market":   clob_token_id,
        "startTs":  start_time_s,
        "endTs":    end_time_s,
        "fidelity": 5   # 5 = 5‐minuters upprälsning
    }

    try:
        r = httpx.get(url, params=params, timeout=10.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[fetch_5m_values_for_token] Fel GET /prices-history för token {clob_token_id}: {e}")
        return []

    history = data.get("history", [])
    if not isinstance(history, list):
        return []

    # Extrahera alla 'p'-värden
    p_vals = [point["p"] for point in history if "p" in point]
    return p_vals  # kan vara tom lista om inget matchar


def main():
    # Tre condition_id för utvalda BTC-relaterade marknader
    cond_id_BTC1 = "0xd8b9ff369452daebce1ac8cb6a29d6817903e85168356c72812317f38e317613"  # Will Bitcoin reach $1,000,000 by Dec 31, 2025?
    cond_id_BTC2 = "0x80026f98f9de40aea8dba02798c4f0067942bba401fa3715209ee7c27482640b"  # US National BTC reserve in 2025?
    cond_id_BTC3 = "0x19ee98e348c0ccb341d1b9566fa14521566e9b2ea7aed34dc407a0ec56be36a2"  # MicroStrategy sells any BTC in 2025?

    condition_ids = [cond_id_BTC1, cond_id_BTC2, cond_id_BTC3]

    # Beräkna tidsfönster: senaste 48 timmar (i sekunder)
    now_s = int(time.time())
    window_s = 48 * 60 * 60  # 48 timmar i sekunder
    start_s = now_s - window_s
    end_s = now_s

    # Samla alla p-värden från alla marknader
    all_p_values = []

    for cond_id in condition_ids:
        print(f"\n=== Processing condition_id={cond_id} ===")
        clob_tokens = get_clob_tokens_for_condition(cond_id)
        if not clob_tokens:
            print(f"Kunde inte hämta några clobTokenIds för condition {cond_id}. Hoppar över.")
            continue

        # Vi använder bara första token_id i listan för testet
        token_id = clob_tokens[0]
        print(f"Använder clob_token_id = {token_id}")

        # Hämta listan av p-värden för detta token under önskat tidsfönster
        p_vals = fetch_5m_values_for_token(token_id, start_s, end_s)
        if not p_vals:
            print(f"Inga p-värden hittades för token {token_id} i intervallet {start_s}–{end_s}.")
            continue

        print(f"Antal p-värden för token {token_id}: {len(p_vals)}")
        print(f"Exempel på p-värden (första 5): {p_vals[:5]}")
        all_p_values.extend(p_vals)

    # Nu har vi en “flat” lista all_p_values av alla p från alla marknader
    if not all_p_values:
        sentiment_score = 0.5  # neutralt fallback
        print("\nInga p-värden samlade från någon marknad; sätter sentiment_score = 0.5")
    else:
        sentiment_score = float(np.mean(all_p_values))
        print(f"\nTotalt antal p-värden insamlade: {len(all_p_values)}")
        print(f"Medelvärde av alla p-värden → sentiment_score = {sentiment_score:.5f}")

    # Om du vill omvandla till [-1, +1]-skala:
    sentiment_signed = 2 * sentiment_score - 1
    print(f"Omvandlat till [–1,+1]-skala → {sentiment_signed:.5f}")

if __name__ == "__main__":
    main()
