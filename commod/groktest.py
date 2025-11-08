#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to fetch historical BID_ASK and TRADES ticks for gold and silver futures using ib_insync,
paginate over the date range, and store in a single NPZ file with separate structures for each symbol.
Based on the approach outlined in hist.py, but with pagination, timezone handling, and NPZ persistence.

Assumes:
- Gold: GCZ5 (December 2025 futures on COMEX)
- Silver: SIZ5 (December 2025 futures on COMEX)
- Date range: November 6, 2025, 09:00:00 to 16:00:00 US/Eastern (converted to UTC for IB)
- Requires ib_insync library (pip install ib_insync)
- IB TWS or Gateway running on localhost:7497
- Appropriate market data subscriptions

Output NPZ keys:
- gold_quotes, gold_trades, gold_meta
- silver_quotes, silver_trades, silver_meta
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import time

from ib_insync import IB, Contract, util

# =====================
# Time handling
# =====================

def parse_et_to_utc_epoch(dt_str: str, fmt: str = "%Y%m%d %H:%M:%S") -> int:
    """Parse datetime string as US/Eastern and return UTC epoch seconds."""
    dt = datetime.strptime(dt_str, fmt)
    dt_et = dt.replace(tzinfo=ZoneInfo("America/New_York"))
    dt_utc = dt_et.astimezone(timezone.utc)
    return int(dt_utc.timestamp())

def ib_utc_str_from_epoch(ts: int) -> str:
    """Format UTC epoch seconds to IB string with ' UTC' suffix."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d %H:%M:%S UTC")

# =====================
# Fetching with pagination
# =====================

def fetch_all_ticks(ib: IB, contract: Contract, what: str, start_utc: int, end_utc: int,
                    use_rth: bool = False, max_per_call: int = 1000, throttle_s: float = 0.1):
    """Paginate reqHistoricalTicks for BID_ASK or TRADES over the time window."""
    recs = []
    cur_start = start_utc
    stalls = 0
    max_stalls = 5

    while cur_start < end_utc:
        start_str = ib_utc_str_from_epoch(cur_start)
        end_str = ib_utc_str_from_epoch(end_utc)
        ticks = ib.reqHistoricalTicks(contract, start_str, end_str, max_per_call, what, use_rth, True, [])

        if not ticks:
            break

        page_last_ts = None
        for t in ticks:
            ts = int(t.time.timestamp())
            if what == "BID_ASK":
                recs.append((ts, t.priceBid, t.priceAsk, t.sizeBid, t.sizeAsk))
            elif what == "TRADES":
                recs.append((ts, t.price, t.size))
            page_last_ts = ts if (page_last_ts is None or ts > page_last_ts) else page_last_ts

        print(f"{contract.symbol} {what} page: start={cur_start} last={page_last_ts} count={len(ticks)}")

        if page_last_ts is None:
            break

        if page_last_ts <= cur_start:
            stalls += 1
            if stalls >= max_stalls:
                print(f"Max stalls reached for {contract.symbol} {what}")
                break
            cur_start += 1  # nudge forward
        else:
            stalls = 0
            cur_start = page_last_ts + 1

        time.sleep(throttle_s)

    return recs

def ticks_to_bidask_array(recs):
    if not recs:
        return np.empty(0, dtype=[("time", "i8"), ("bid", "f8"), ("ask", "f8"), ("bid_size", "i8"), ("ask_size", "i8")])
    return np.array(recs, dtype=[("time", "i8"), ("bid", "f8"), ("ask", "f8"), ("bid_size", "i8"), ("ask_size", "i8")])

def ticks_to_trades_array(recs):
    if not recs:
        return np.empty(0, dtype=[("time", "i8"), ("price", "f8"), ("size", "i8")])
    return np.array(recs, dtype=[("time", "i8"), ("price", "f8"), ("size", "i8")])

# =====================
# Main
# =====================

if __name__ == "__main__":
    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=23132213, readonly=True, timeout=3.0)

    # Define contracts (December 2025 futures)
    gold_contract = Contract(symbol="GC", secType="FUT", exchange="COMEX", currency="USD", lastTradeDateOrContractMonth="202512")
    silver_contract = Contract(symbol="SI", secType="FUT", exchange="COMEX", currency="USD", lastTradeDateOrContractMonth="202512")

    # Qualify contracts
    qualified = ib.qualifyContracts(gold_contract, silver_contract)
    if len(qualified) != 2:
        raise RuntimeError("Failed to qualify contracts")

    # Date range (as in hist.py, interpreted as US/Eastern)
    start_et_str = "20251106 09:00:00"
    end_et_str = "20251106 16:00:00"
    start_utc = parse_et_to_utc_epoch(start_et_str)
    end_utc = parse_et_to_utc_epoch(end_et_str)

    # Fetch gold
    print(f"Fetching GOLD BID_ASK {ib_utc_str_from_epoch(start_utc)} -> {ib_utc_str_from_epoch(end_utc)}")
    gold_bidask_recs = fetch_all_ticks(ib, gold_contract, "BID_ASK", start_utc, end_utc)
    gold_quotes = ticks_to_bidask_array(gold_bidask_recs)
    print(f"Fetched {gold_quotes.size} GOLD BID_ASK ticks")

    print("Fetching GOLD TRADES")
    gold_trades_recs = fetch_all_ticks(ib, gold_contract, "TRADES", start_utc, end_utc)
    gold_trades = ticks_to_trades_array(gold_trades_recs)
    print(f"Fetched {gold_trades.size} GOLD TRADE ticks")

    gold_meta = {
        "symbol": "GC",
        "start_et": start_et_str,
        "end_et": end_et_str,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "input_tz": "US/Eastern",
        "ib_tz": "UTC"
    }

    # Fetch silver
    print(f"Fetching SILVER BID_ASK {ib_utc_str_from_epoch(start_utc)} -> {ib_utc_str_from_epoch(end_utc)}")
    silver_bidask_recs = fetch_all_ticks(ib, silver_contract, "BID_ASK", start_utc, end_utc)
    silver_quotes = ticks_to_bidask_array(silver_bidask_recs)
    print(f"Fetched {silver_quotes.size} SILVER BID_ASK ticks")

    print("Fetching SILVER TRADES")
    silver_trades_recs = fetch_all_ticks(ib, silver_contract, "TRADES", start_utc, end_utc)
    silver_trades = ticks_to_trades_array(silver_trades_recs)
    print(f"Fetched {silver_trades.size} SILVER TRADE ticks")

    silver_meta = {
        "symbol": "SI",
        "start_et": start_et_str,
        "end_et": end_et_str,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "input_tz": "US/Eastern",
        "ib_tz": "UTC"
    }

    # Save to NPZ
    out_path = "gold_silver_ticks.npz"
    np.savez_compressed(
        out_path,
        gold_quotes=gold_quotes,
        gold_trades=gold_trades,
        gold_meta=np.array([gold_meta], dtype=object),
        silver_quotes=silver_quotes,
        silver_trades=silver_trades,
        silver_meta=np.array([silver_meta], dtype=object)
    )
    print(f"Saved to {out_path}")

    ib.disconnect()
