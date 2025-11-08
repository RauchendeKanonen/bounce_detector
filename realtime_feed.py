#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ib_realtime_feed_to_sampler.py

Live IBKR feed → RealtimeSampler bridge
- Subscribes to Tick-By-Tick streams: BidAsk + AllLast (trades)
- Feeds ticks into the provided RealtimeSampler (1s bins by default)
- Thread-safe, graceful shutdown, light pacing handling
- Command line uses German date format in **US/Eastern** for optional runtime windowing (start/end) if you want to run only during RTH.
- Optional CSV export of the sampler buffer for quick inspection.

Requirements:
  pip install ibapi

Usage examples:
  # Start live sampling for AAPL, 1s bins, keep last 3600 bins (~1h)
  python ib_realtime_feed_to_sampler.py \
    --host 127.0.0.1 --port 7497 --client-id 1002 \
    --symbol AAPL --exchange SMART --currency USD \
    --dt-sec 1 --buffer-bins 3600 \
    --export-csv aapl_live_snapshot.csv

  # With an optional time gate in US/Eastern using German format
  python ib_realtime_feed_to_sampler.py \
    --symbol AAPL --start-et "28.10.2024 09:30:00" --end-et "28.10.2024 16:00:00"

Notes:
- IB timestamps for these callbacks are in seconds. For sub-second ordering, you can add a recv_ns in the sampler if needed.
- Ensure you have live market data permissions for the instruments.
- For FX or Futures, adjust contract builder accordingly.
"""

import argparse
import signal
import sys
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# ---- import the sampler from your file or paste it here ----
#from realtime_sampler import RealtimeSampler, SAMPLED_DTYPE
from BounceModel import create_bouncedetector
# Minimal inline copy of the class name to avoid NameError in this standalone file.
# Replace this with `from realtime_sampler import RealtimeSampler` in your project.
from math import log
import numpy as np
from collections import deque

from rt_plot import RealtimeGraph
import time, random

# ===== Structured dtype (same as offline sampler) =====
SAMPLED_DTYPE = [
    ("t_left", "i8"), ("t_right", "i8"),
    ("bid", "f8"), ("ask", "f8"),
    ("bid_size", "f8"), ("ask_size", "f8"),
    ("last_trade_price", "f8"),
    ("trade_volume", "f8"),
    ("buy_count", "i8"), ("sell_count", "i8"),
    ("updates_count", "i8"),
    ("mid", "f8"), ("spread", "f8"), ("microprice", "f8"),
    ("ret_1", "f8"),
    ("d_bid", "f8"), ("d_ask", "f8"),
    ("d_bid_size", "f8"), ("d_ask_size", "f8"),
    ("flow_intensity", "f8"),
]


class RealtimeSampler:
    def __init__(self, dt_sec: int, buffer_bins: int, include_trades: bool = True, align_to: Optional[int] = None):
        assert dt_sec >= 1 and buffer_bins >= 2
        self.dt = int(dt_sec)
        self.include_trades = include_trades
        self._quotes = deque()  # (ts,bid,ask,bs,as)
        self._trades = deque()  # (ts,price,size)
        self._inq_lock = threading.Lock()
        self._buf_lock = threading.Lock()
        self._N = int(buffer_bins)
        self._buf = np.zeros(self._N, dtype=SAMPLED_DTYPE)
        self._buf.fill(0)
        for k in ("buy_count","sell_count","updates_count"):
            self._buf[k] = 0
        self._widx = 0
        self._bins_written = 0
        self._prev_bid = np.nan
        self._prev_ask = np.nan
        self._prev_bid_size = np.nan
        self._prev_ask_size = np.nan
        self._prev_mid = np.nan
        t0 = int(time.time()) if align_to is None else int(align_to)
        self._cur_bin_left = (t0 // self.dt) * self.dt
        self._cur_bin_right = self._cur_bin_left + self.dt
        self._stop_evt = threading.Event()
        self._thr = None

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop_evt.clear()
        self._thr = threading.Thread(target=self._run_loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop_evt.set()
        if self._thr:
            self._thr.join(timeout=2.0)

    def push_quote(self, ts_sec: int, bid: float, ask: float, bid_size: int, ask_size: int):
        with self._inq_lock:
            self._quotes.append((int(ts_sec), float(bid), float(ask), int(bid_size), int(ask_size)))

    def push_trade(self, ts_sec: int, price: float, size: int):
        if not self.include_trades:
            return
        with self._inq_lock:
            self._trades.append((int(ts_sec), float(price), int(size)))

    def save_npz(self, path: str):
        with self._buf_lock:
            n = min(self._bins_written, self._N)
            if n <= 0:
                arr = self._buf[:0].copy()
            else:
                end = (self._widx - 1) % self._N
                start = (end - (n - 1)) % self._N
                if start <= end:
                    arr = self._buf[start:end+1].copy()
                else:
                    arr = np.concatenate([self._buf[start:], self._buf[:end+1]], axis=0).copy()
        meta = dict(ts=time.time(), dt_sec=self.dt, n_bins=int(arr.shape[0]), live=True)
        np.savez_compressed(path, sampled=arr, info=np.array([meta], dtype=object))

    def get_recent_window(self, n_bins: int, fields=None):
        with self._buf_lock:
            n = min(n_bins, min(self._bins_written, self._N))
            if n <= 0:
                return np.empty(0, dtype=self._buf.dtype)
            end = (self._widx - 1) % self._N
            start = (end - (n - 1)) % self._N
            if start <= end:
                view = self._buf[start:end+1].copy()
            else:
                view = np.concatenate([self._buf[start:], self._buf[:end+1]], axis=0).copy()
        if fields:
            return np.column_stack([view[f] for f in fields])
        return view

    def _run_loop(self):
        poll = 0.02
        while not self._stop_evt.is_set():
            now = int(time.time())
            while self._cur_bin_right <= now:
                q_list, t_list = self._drain_until(self._cur_bin_right - 1)
                self._commit_bin(self._cur_bin_left, self._cur_bin_right, q_list, t_list)
                self._cur_bin_left = self._cur_bin_right
                self._cur_bin_right = self._cur_bin_left + self.dt
            time.sleep(poll)

    def _drain_until(self, cutoff_sec: int):
        with self._inq_lock:
            q_out = []
            while self._quotes and self._quotes[0][0] <= cutoff_sec:
                q_out.append(self._quotes.popleft())
            t_out = []
            while self._trades and self._trades[0][0] <= cutoff_sec:
                t_out.append(self._trades.popleft())
        return q_out, t_out

    def _commit_bin(self, t_left: int, t_right: int, q_list, t_list):
        updates_count = len(q_list)
        bid, ask, bid_size, ask_size = self._prev_bid, self._prev_ask, self._prev_bid_size, self._prev_ask_size
        if updates_count:
            _, bid, ask, bid_size, ask_size = q_list[-1]
        last_trade_price = np.nan
        trade_volume = 0.0
        buy_count = 0
        sell_count = 0
        if t_list:
            last_trade_price = t_list[-1][1]
            trade_volume = float(sum(x[2] for x in t_list))
            pbid, pask = bid, ask
            if not np.isnan(pbid) and not np.isnan(pask):
                for (_, price, _sz) in t_list:
                    if price >= pask:
                        buy_count += 1
                    elif price <= pbid:
                        sell_count += 1
        mid = (bid + ask) * 0.5 if (not np.isnan(bid) and not np.isnan(ask)) else np.nan
        spread = (ask - bid) if (not np.isnan(bid) and not np.isnan(ask)) else np.nan
        microprice = np.nan
        denom = (bid_size + ask_size) if (not np.isnan(bid_size) and not np.isnan(ask_size)) else np.nan
        if denom and denom != 0 and not np.isnan(bid) and not np.isnan(ask):
            microprice = (ask * bid_size + bid * ask_size) / denom
        d_bid = np.nan if np.isnan(self._prev_bid) or np.isnan(bid) else (bid - self._prev_bid)
        d_ask = np.nan if np.isnan(self._prev_ask) or np.isnan(ask) else (ask - self._prev_ask)
        d_bid_size = np.nan if np.isnan(self._prev_bid_size) or np.isnan(bid_size) else (bid_size - self._prev_bid_size)
        d_ask_size = np.nan if np.isnan(self._prev_ask_size) or np.isnan(ask_size) else (ask_size - self._prev_ask_size)
        ret_1 = np.nan
        if not np.isnan(mid) and not np.isnan(self._prev_mid) and self._prev_mid > 0 and mid > 0:
            ret_1 = log(mid) - log(self._prev_mid)
        flow_intensity = float(buy_count - sell_count)
        row = np.zeros((), dtype=SAMPLED_DTYPE)
        row["t_left"], row["t_right"] = int(t_left), int(t_right)
        row["bid"], row["ask"], row["bid_size"], row["ask_size"] = bid, ask, bid_size, ask_size
        row["last_trade_price"], row["trade_volume"] = last_trade_price, trade_volume
        row["buy_count"], row["sell_count"], row["updates_count"] = int(buy_count), int(sell_count), int(updates_count)
        row["mid"], row["spread"], row["microprice"] = mid, spread, microprice
        row["ret_1"], row["d_bid"], row["d_ask"], row["d_bid_size"], row["d_ask_size"] = ret_1, d_bid, d_ask, d_bid_size, d_ask_size
        row["flow_intensity"] = flow_intensity
        with self._buf_lock:
            self._buf[self._widx] = row
            self._widx = (self._widx + 1) % self._N
            self._bins_written += 1
        self._prev_bid, self._prev_ask, self._prev_bid_size, self._prev_ask_size, self._prev_mid = bid, ask, bid_size, ask_size, mid

# ---------------- IB wiring ----------------

class IBFeed(EWrapper, EClient):
    def __init__(self, sampler: RealtimeSampler):
        EClient.__init__(self, self)
        self.sampler = sampler
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._subs = {}
        self._stop = False

    # Lifecycle
    def nextValidId(self, orderId: int):
        self._ready.set()

    def connect_and_start(self, host: str, port: int, client_id: int):
        self.connect(host, port, clientId=client_id)
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("IB connection not ready in time")

    def disconnect_clean(self):
        with self._lock:
            # cancel all tick-by-tick subs
            for rid in list(self._subs.keys()):
                try:
                    self.cancelTickByTickData(rid)
                except Exception:
                    pass
            self._subs.clear()
        try:
            self.disconnect()
        except Exception:
            pass

    # Errors
    def error(self, reqId, code, msg, advancedOrderRejectJson=""):
        # Handle common pacing codes: 162, 354, 10197
        if code in (162, 354, 10197):
            # Light backoff
            time.sleep(0.5)
        else:
            # Print other errors for debugging
            sys.stderr.write(f"IB error reqId={reqId} code={code} msg={msg}\n")

    # ---- Tick-by-tick callbacks ----
    # Trades (AllLast or Last)
    def tickByTickAllLast(self, reqId, tickType, time_ts, price, size, tickAttribLast, exchange, specialConditions):
        # time_ts is epoch seconds
        self.sampler.push_trade(time_ts, price, size)

    # Bid/Ask top of book
    def tickByTickBidAsk(self, reqId, time_ts, bidPrice, askPrice, bidSize, askSize, tickAttribBidAsk):
        self.sampler.push_quote(time_ts, bidPrice, askPrice, bidSize, askSize)

# ---------------- Helpers ----------------

def make_stock(symbol: str, currency: str = "USD", exchange: str = "SMART") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    return c


def parse_german_et(s: Optional[str]) -> Optional[int]:
    """Parse German date/time string in US/Eastern, return epoch seconds (UTC-based int)."""
    if not s:
        return None
    s = s.strip()
    fmts = ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y"]
    tz = ZoneInfo("America/New_York")
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            dt = dt.replace(tzinfo=tz)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date/time: {s}")


def prepare_window_struct(window_struct, bd):
    """
    window_struct: structured np.ndarray with dtype.names like
      ('t_left','t_right','bid','ask','bid_size','ask_size', ...)

    Returns: np.ndarray of shape [T, 1, F] in the SAME order as bd.feature_fields
    """
    # 1) make sure the window length matches model expectation
    T_needed = bd.window_len
    if len(window_struct) != T_needed:
        raise ValueError(f"Need T={T_needed} rows, got {len(window_struct)}")

    # 2) verify features exist in the structured dtype
    names = set(window_struct.dtype.names or [])
    required = bd.feature_fields
    missing = [n for n in required if n not in names]
    if missing:
        raise KeyError(f"Missing required feature(s) in window: {missing}\n"
                       f"Available: {sorted(names)}\n"
                       f"Expected:  {required}")

    # 3) stack columns in the saved training order, cast to float32
    X = np.column_stack([window_struct[name] for name in required]).astype(np.float32, copy=False)  # [T, F]

    # 4) add stream axis S=1  -> [T, 1, F]
    X = X[:, None, :]
    return X
# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="IBKR live feed → RealtimeSampler")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=1002)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--currency", default="USD")

    ap.add_argument("--dt-sec", type=int, default=1)
    ap.add_argument("--buffer-bins", type=int, default=3600)

    ap.add_argument("--start-et", help="Start (German format) interpreted in US/Eastern, e.g. '28.10.2024 09:30:00'")
    ap.add_argument("--end-et", help="End   (German format) interpreted in US/Eastern, e.g. '28.10.2024 16:00:00'")

    ap.add_argument("--export-npz", help="Optional NPZ export of the rolling buffer on exit")

    ap.add_argument("--no-trades", action="store_true", help="Disable trade stream (quotes only)")
    ap.add_argument("--rt-plot", action="store_true", help="Plot on a canvas")


    args = ap.parse_args()

    sampler = RealtimeSampler(dt_sec=args.dt_sec, buffer_bins=args.buffer_bins, include_trades=not args.no_trades)
    sampler.start()

    app = IBFeed(sampler)
    app.connect_and_start(args.host, args.port, args.client_id)

    # build contract
    contract = make_stock(args.symbol, args.currency, args.exchange)

    # optional time window in US/Eastern (German input)
    start_ts = parse_german_et(args.start_et) if args.start_et else None
    end_ts = parse_german_et(args.end_et) if args.end_et else None
    Plot = None
    if args.rt_plot:
        Plot = RealtimeGraph(title=args.symbol, tzname="America/New_York", max_points=100_000)
        Plot.add_series("Bid/Ask", axis="left")
        Plot.add_series("Probability", axis="right")
        Plot.add_series("Prediction", axis="right")
        Plot.set_labels(left="$", right="1")
        Plot.set_x_range(seconds=300)  # rolling 5-minute window



    bd = create_bouncedetector("models/bounce.ckpt", device="cuda")  # or "cpu"


    # subscribe tick-by-tick
    # We use dedicated reqIds: 10001 (AllLast), 10002 (BidAsk)
    try:
        if not args.no_trades:
            app.reqTickByTickData(10001, contract, "AllLast", 0, False)
        app.reqTickByTickData(10002, contract, "BidAsk", 0, False)
    except Exception as e:
        sys.stderr.write(f"Subscription error: {e}\n")

    # Gate by optional start/end times
    def inside_window() -> bool:
        now = int(time.time())
        if start_ts is not None and now < start_ts:
            return False
        if end_ts is not None and now >= end_ts:
            return False
        return True

    stop_flag = threading.Event()

    def handle_sig(signum, frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # Main supervision loop
    try:
        while not stop_flag.is_set():
            if not inside_window():
                if end_ts is not None and int(time.time()) >= end_ts:
                    break
                time.sleep(0.25)
                continue
            time.sleep(0.25)
            window = sampler.get_recent_window(bd.window_len)
            print(sampler.get_recent_window(1))
            if len(window) >= bd.window_len:
                window = prepare_window_struct(window, bd)

                prob = bd.predict_window(window)
                print(window[-1])
                yhat = (prob >= 0.5)
                print(prob)
                if Plot is not None:
                    t = time.time()  # Unix seconds
                    i_bid = bd.feature_fields.index('bid')
                    i_ask = bd.feature_fields.index('ask')

                    # most recent for stream 0
                    bid = float(window[-1, 0, i_bid])
                    ask = float(window[-1, 0, i_ask])

                    Plot.add_point("Bid/Ask", t, (bid+ask)/2)
                    Plot.add_point("Probability", t, prob)
                    Plot.add_point("Prediction", t, yhat)
                    #time.sleep(0.05)  # 20 Hz
    finally:
        # Clean down
        try:
            if args.export_npz:
                arr = sampler.get_recent_window(args.buffer_bins)
                np.savez_compressed(args.export_npz, sampled=arr)
                print(f"Saved NPZ → {args.export_npz}")
        except Exception as e:
            sys.stderr.write(f"CSV export error: {e}\n")
        try:
            # Cancel subs
            app.cancelTickByTickData(10002)
            if not args.no_trades:
                app.cancelTickByTickData(10001)
        except Exception:
            pass
        try:
            app.disconnect_clean()
        finally:
            sampler.stop()


if __name__ == "__main__":
    main()
