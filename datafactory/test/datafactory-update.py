#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ib_ticks_pipeline_et.py

- fetch  : download raw ticks
- sample : bin into Δt features
- update : **add XAUUSD / XAGUSD (or any symbol) to an existing file**
          using the file's original start/end time.
"""

import math
import time
import queue
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# =====================
# Time / formatting
# =====================
from datetime import datetime, timezone
DE_FMT_FULL = "%d.%m.%Y %H:%M:%S"
DE_FMT_DATE = "%d.%m.%Y"
from zoneinfo import ZoneInfo

def parse_german_us_eastern(dt_str: str) -> int:
    dt_str = dt_str.strip()
    if not dt_str:
        raise ValueError("Empty datetime string")
    try:
        if len(dt_str) == 10:
            dt = datetime.strptime(dt_str, DE_FMT_DATE)
            dt = dt.replace(hour=0, minute=0, second=0)
        else:
            dt = datetime.strptime(dt_str, DE_FMT_FULL)
    except ValueError as e:
        raise ValueError(
            f"Invalid date/time '{dt_str}'. Expected 'DD.MM.YYYY HH:MM:SS' or 'DD.MM.YYYY'."
        ) from e
    dt_et = dt.replace(tzinfo=ZoneInfo("America/New_York"))
    dt_utc = dt_et.astimezone(timezone.utc)
    return int(dt_utc.timestamp())

def ib_utc_str_from_epoch(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y%m%d %H:%M:%S") + " UTC"


# =======================
# IB client/wrapper
# =======================
class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self._next_req_id = 1
        self.lock = threading.Lock()
        self.result_queues: Dict[int, queue.Queue] = {}
        self.done_flags: Dict[int, bool] = {}
        self.connected_event = threading.Event()
        self.disconnected_event = threading.Event()

    def connect_and_start(self, host="127.0.0.1", port=7497, client_id=1001):
        self.connect(host, port, clientId=client_id)
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        self.connected_event.wait(5.0)

    def nextValidId(self, orderId: int):
        self.connected_event.set()

    def connectionClosed(self):
        self.disconnected_event.set()

    def _alloc_req(self) -> int:
        with self.lock:
            rid = self._next_req_id
            self._next_req_id += 1
            self.result_queues[rid] = queue.Queue()
            self.done_flags[rid] = False
            return rid

    def _finish_req(self, reqId: int):
        self.done_flags[reqId] = True

    def historicalTicksBidAsk(self, reqId, ticks, done):
        q = self.result_queues.get(reqId)
        if q:
            for t in ticks:
                q.put(("BID_ASK", int(t.time), float(t.priceBid), float(t.priceAsk),
                       int(t.sizeBid), int(t.sizeAsk)))
        if done:
            self._finish_req(reqId)

    def historicalTicksLast(self, reqId, ticks, done):
        q = self.result_queues.get(reqId)
        if q:
            for t in ticks:
                q.put(("TRADES", int(t.time), float(t.price), int(t.size)))
        if done:
            self._finish_req(reqId)

    def historicalTicks(self, reqId, ticks, done):
        if done:
            self._finish_req(reqId)

    def error(self, reqId, code, msg, advancedOrderRejectJson=""):
        q = self.result_queues.get(reqId)
        if q:
            q.put(("ERROR", code, msg))


# =========================
# Fetching / Pagination
# =========================
@dataclass
class FetchWindow:
    start_utc: int
    end_utc: int


def _paginate_ticks(
    app: IBApp,
    contract: Contract,
    what: str,
    start_utc: int,
    end_utc: int,
    rth_only: int,
    ignore_size: bool,
    max_per_call: int = 1000,
    throttle_s: float = 0.1,
    req_timeout: float = 30.0,
):
    cur = int(start_utc)
    end = int(end_utc)
    stalls = 0
    MAX_STALLS = 5

    while cur < end:
        reqId = app._alloc_req()
        app.reqHistoricalTicks(
            reqId=reqId,
            contract=contract,
            startDateTime=ib_utc_str_from_epoch(cur),
            endDateTime=ib_utc_str_from_epoch(end),
            numberOfTicks=max_per_call,
            whatToShow=what,
            useRth=rth_only,
            ignoreSize=ignore_size,
            miscOptions=[]
        )

        q = app.result_queues[reqId]
        page_last = None
        n_recs = 0
        start_time = time.time()

        while True:
            if time.time() - start_time > req_timeout:
                print(f"{what} request {reqId} TIMED OUT – re-issuing")
                app.cancelHistoricalData(reqId)
                with app.lock:
                    app.result_queues.pop(reqId, None)
                    app.done_flags.pop(reqId, None)
                break

            if app.done_flags.get(reqId, False) and q.empty():
                break

            try:
                item = q.get(timeout=0.2)
            except queue.Empty:
                continue

            typ = item[0]
            if typ == "ERROR":
                _, code, msg = item
                if code in (162, 354, 10197):
                    time.sleep(max(throttle_s * 10, 1.0))
                    raise RuntimeError(f"Pacing error ({what}) code={code} msg={msg}")
                raise RuntimeError(f"IB error ({what}) reqId={reqId} code={code} msg={msg}")

            if what == "BID_ASK" and typ == "BID_ASK":
                yield item
                ts = item[1]
                page_last = ts if (page_last is None or ts > page_last) else page_last
                n_recs += 1
            elif what == "TRADES" and typ == "TRADES":
                yield item
                ts = item[1]
                page_last = ts if (page_last is None or ts > page_last) else page_last
                n_recs += 1

        print(f"{what} page: start={cur} last={page_last} count={n_recs}")

        if page_last is None:
            break

        if page_last <= cur:
            stalls += 1
            print("stall")
        else:
            stalls = 0
            cur = int(page_last) + 1

        time.sleep(throttle_s)


def fetch_historical_bidask(
    app: IBApp, contract: Contract, win: FetchWindow,
    rth_only: int = 0, ignore_size: bool = False,
    max_per_call: int = 1000, throttle_s: float = 0.1,
    req_timeout: float = 30.0,
) -> np.ndarray:
    recs: List[Tuple[int, float, float, int, int]] = []
    for item in _paginate_ticks(
        app, contract, "BID_ASK", win.start_utc, win.end_utc,
        rth_only, ignore_size, max_per_call, throttle_s, req_timeout
    ):
        _, ts, bid, ask, bs, aS = item
        recs.append((ts, bid, ask, bs, aS))

    if not recs:
        return np.empty(0, dtype=[("time","i8"),("bid","f8"),("ask","f8"),
                                 ("bid_size","i8"),("ask_size","i8")])
    return np.array(recs, dtype=[("time","i8"),("bid","f8"),("ask","f8"),
                                 ("bid_size","i8"),("ask_size","i8")])

def fetch_historical_trades(
    app: IBApp, contract: Contract, win: FetchWindow,
    rth_only: int = 0, max_per_call: int = 1000,
    throttle_s: float = 0.1, req_timeout: float = 30.0,
) -> np.ndarray:
    recs: List[Tuple[int, float, int]] = []
    for item in _paginate_ticks(
        app, contract, "TRADES", win.start_utc, win.end_utc,
        rth_only, False, max_per_call, throttle_s, req_timeout
    ):
        _, ts, price, size = item
        recs.append((ts, price, size))

    if not recs:
        return np.empty(0, dtype=[("time","i8"),("price","f8"),("size","i8")])
    return np.array(recs, dtype=[("time","i8"),("price","f8"),("size","i8")])


# =========================
# Persistence helpers
# =========================
def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    meta_obj = data["meta"][0]
    meta = dict(meta_obj) if isinstance(meta_obj, dict) else meta_obj.item()

    symbols = meta.get("symbols", None)
    if symbols and len(symbols) > 1:
        all_quotes = {sym: data[f"quotes_{sym}"] for sym in symbols}
        all_trades = {sym: data[f"trades_{sym}"] for sym in symbols}
    else:
        all_quotes = {"DEFAULT": data["quotes"]}
        all_trades = {"DEFAULT": data["trades"]}
        symbols = ["DEFAULT"]
        meta["symbols"] = symbols
    return all_quotes, all_trades, meta


# =========================
# Sampling & features
# =========================
def make_bins(start_utc: int, end_utc: int, dt_sec: int) -> np.ndarray:
    n = max(0, math.ceil((end_utc - start_utc) / dt_sec))
    edges = start_utc + np.arange(n + 1, dtype=np.int64) * dt_sec
    return edges


def sample_to_bins(
    quotes: np.ndarray,
    trades: np.ndarray,
    dt_sec: int,
    include_trades: bool = True,
    edges: Optional[np.ndarray] = None,
):
    if quotes.size == 0:
        raise ValueError("No quotes provided; cannot build snapshots.")

    t0 = int(quotes["time"][0])
    t1 = int(quotes["time"][-1])
    if trades.size:
        t1 = max(t1, int(trades["time"][-1]))

    if edges is None:
        edges = make_bins(t0, t1 + dt_sec, dt_sec)

    bin_left = edges[:-1]
    bin_right = edges[1:]

    q_times = quotes["time"].astype(np.int64)
    idx_last = np.searchsorted(q_times, bin_right - 1, side="right") - 1
    valid_mask = idx_last >= 0

    bid = np.full(bin_left.shape, np.nan)
    ask = np.full(bin_left.shape, np.nan)
    bid_size = np.full(bin_left.shape, np.nan)
    ask_size = np.full(bin_left.shape, np.nan)
    updates_count = np.zeros(bin_left.shape, dtype=np.int64)

    bid[valid_mask] = quotes["bid"][idx_last[valid_mask]]
    ask[valid_mask] = quotes["ask"][idx_last[valid_mask]]
    bid_size[valid_mask] = quotes["bid_size"][idx_last[valid_mask]]
    ask_size[valid_mask] = quotes["ask_size"][idx_last[valid_mask]]

    q_bins = np.searchsorted(edges, q_times, side="right") - 1
    q_bins = np.clip(q_bins, 0, len(bin_left)-1)
    updates_count += np.bincount(q_bins, minlength=len(bin_left))

    last_trade_price = np.full(bin_left.shape, np.nan)
    trade_volume = np.zeros(bin_left.shape, dtype=np.float64)
    buy_count = np.zeros(bin_left.shape, dtype=np.int64)
    sell_count = np.zeros(bin_left.shape, dtype=np.int64)

    if include_trades and trades.size:
        tr_times = trades["time"].astype(np.int64)
        tr_bins = np.searchsorted(edges, tr_times, side="right") - 1

        last_trade_idx: Dict[int, int] = {}
        for i, b in enumerate(tr_bins):
            if 0 <= b < len(bin_left):
                prev = last_trade_idx.get(b)
                if prev is None or tr_times[i] >= tr_times[prev]:
                    last_trade_idx[b] = i
                trade_volume[b] += float(trades["size"][i])

        for b, i in last_trade_idx.items():
            last_trade_price[b] = trades["price"][i]

        prev_q_idx = np.searchsorted(q_times, tr_times, side="right") - 1
        valid_tr = prev_q_idx >= 0
        if np.any(valid_tr):
            pbid = quotes["bid"][prev_q_idx[valid_tr]]
            pask = quotes["ask"][prev_q_idx[valid_tr]]
            tprice = trades["price"][valid_tr]
            tbins_valid = tr_bins[valid_tr]

            buys = tprice >= pask
            sells = tprice <= pbid
            for j in range(buys.size):
                b = tbins_valid[j]
                if 0 <= b < len(bin_left):
                    if buys[j]:
                        buy_count[b] += 1
                    elif sells[j]:
                        sell_count[b] += 1

    mid = (bid + ask) / 2.0
    spread = (ask - bid)
    with np.errstate(divide="ignore", invalid="ignore"):
        microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

    ret_1 = np.full(bin_left.shape, np.nan)
    valid_mid = ~np.isnan(mid)
    for k in range(1, len(mid)):
        if valid_mid[k] and valid_mid[k-1]:
            ret_1[k] = math.log(mid[k]) - math.log(mid[k-1])

    def delta(arr):
        d = np.full(arr.shape, np.nan)
        d[1:] = arr[1:] - arr[:-1]
        return d

    d_bid = delta(bid)
    d_ask = delta(ask)
    d_bid_size = delta(bid_size)
    d_ask_size = delta(ask_size)

    dtype = [
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
    out = np.zeros(len(bin_left), dtype=dtype)
    out["t_left"] = bin_left
    out["t_right"] = bin_right
    out["bid"] = bid
    out["ask"] = ask
    out["bid_size"] = bid_size
    out["ask_size"] = ask_size
    out["last_trade_price"] = last_trade_price
    out["trade_volume"] = trade_volume
    out["buy_count"] = buy_count
    out["sell_count"] = sell_count
    out["updates_count"] = updates_count
    out["mid"] = mid
    out["spread"] = spread
    out["microprice"] = microprice
    out["ret_1"] = ret_1
    out["d_bid"] = d_bid
    out["d_ask"] = d_ask
    out["d_bid_size"] = d_bid_size
    out["d_ask_size"] = d_ask_size
    out["flow_intensity"] = (buy_count - sell_count).astype(float)

    info = dict(dt_sec=dt_sec, n_bins=len(out),
                t0=int(bin_left[0]) if len(bin_left) else None,
                t1=int(bin_right[-1]) if len(bin_right) else None)
    return out, info


# =========================
# Contract helper
# =========================
def make_contract(symbol: str, currency="USD", exchange="SMART") -> Contract:
    c = Contract()
    c.symbol = symbol.upper()
    if c.symbol in ["XAUUSD", "XAGUSD"]:
        c.secType = "CMDTY"
    else:
        c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    return c


# =========================
# UPDATE: add missing symbols (XAUUSD / XAGUSD) to existing file
# =========================
def update_file_with_missing_symbols(
    infile: str,
    app: IBApp,
    add_symbols: List[str],
    rth_only: int,
    currency: str,
    exchange: str,
    req_timeout: float,
):
    print(f"Adding symbols {add_symbols} to {infile} using the file's original time window")

    # Load existing data + meta
    all_quotes, all_trades, meta = load_npz(infile)

    # Extract original time window
    start_epoch = meta["start_epoch"]
    end_epoch   = meta["end_epoch"]
    print(f"Original window: {meta['start_et']} → {meta['end_et']} (UTC {start_epoch} → {end_epoch})")

    # Determine which symbols are already present
    existing = set(meta.get("symbols", []))
    if "DEFAULT" in all_quotes:
        existing = {meta.get("symbol", "DEFAULT")}  # single-symbol legacy

    missing = [s for s in add_symbols if s.upper() not in existing]
    if not missing:
        print("All requested symbols already present – nothing to do.")
        return

    print(f"Fetching missing symbols: {missing}")

    # Fetch missing symbols over the **same** window
    new_quotes = {}
    new_trades = {}
    for sym in missing:
        print(f"  → {sym}")
        contract = make_contract(sym, currency, exchange)
        win = FetchWindow(start_utc=start_epoch, end_utc=end_epoch)
        q = fetch_historical_bidask(app, contract, win, rth_only=rth_only, req_timeout=req_timeout)
        t = fetch_historical_trades(app, contract, win, rth_only=rth_only, req_timeout=req_timeout)
        new_quotes[sym] = q
        new_trades[sym] = t
        print(f"    +{q.size} quotes, +{t.size} trades")

    # Merge into save dict
    save_args = {}

    # 1. Copy existing data

    for sym in existing:
        save_args[f"quotes_{sym}"] = all_quotes[sym]
        save_args[f"trades_{sym}"] = all_trades[sym]

    # 2. Add new symbols (always multi-symbol format)
    for sym in missing:
        save_args[f"quotes_{sym}"] = new_quotes[sym]
        save_args[f"trades_{sym}"] = new_trades[sym]

    # 3. Update meta
    all_syms = list(existing) + missing
    meta["symbols"] = all_syms
    save_args["meta"] = np.array([meta], dtype=object)

    # 4. Overwrite file
    np.savez_compressed(infile, **save_args)
    print(f"Updated {infile} – now contains symbols: {all_syms}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IB ticks → NPZ → sample + update (add XAUUSD/XAGUSD)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- fetch -------------------------------------------------
    p_fetch = sub.add_parser("fetch", help="Download raw ticks")
    p_fetch.add_argument("--host", default="127.0.0.1")
    p_fetch.add_argument("--port", type=int, default=7497)
    p_fetch.add_argument("--client-id", type=int, default=1001)
    p_fetch.add_argument("--symbols", type=str, nargs='+', required=True,
                         help="Symbols (space-separated). Auto-detects STK/CMDTY.")
    p_fetch.add_argument("--currency", default="USD")
    p_fetch.add_argument("--exchange", default="SMART")
    p_fetch.add_argument("--start-et", required=True)
    p_fetch.add_argument("--end-et", required=True)
    p_fetch.add_argument("--rth-only", type=int, default=0)
    p_fetch.add_argument("--req-timeout", type=float, default=30.0)
    p_fetch.add_argument("--out", required=True)

    # ---- sample ------------------------------------------------
    p_sample = sub.add_parser("sample", help="Bin into Δt features")
    p_sample.add_argument("--infile", required=True)
    p_sample.add_argument("--dt-sec", type=int, required=True)
    p_sample.add_argument("--symbols", type=str, nargs='*')
    p_sample.add_argument("--outfile", required=True)

    # ---- update (add missing metals) ---------------------------
    p_update = sub.add_parser("update",
        help="Add XAUUSD / XAGUSD (or any symbol) to an existing .npz using its original time window")
    p_update.add_argument("--infile", required=True, help="Existing .npz file")
    p_update.add_argument("--symbols", type=str, nargs='+', required=True,
                          help="Symbols to add (e.g. XAUUSD XAGUSD)")
    p_update.add_argument("--currency", default="USD")
    p_update.add_argument("--exchange", default="SMART")
    p_update.add_argument("--rth-only", type=int, default=0)
    p_update.add_argument("--req-timeout", type=float, default=30.0)
    p_update.add_argument("--host", default="127.0.0.1")
    p_update.add_argument("--port", type=int, default=7497)
    p_update.add_argument("--client-id", type=int, default=1001)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    if args.cmd == "fetch":
        start_epoch = parse_german_us_eastern(args.start_et)
        end_epoch   = parse_german_us_eastern(args.end_et)
        if end_epoch <= start_epoch:
            raise SystemExit("--end-et must be after --start-et")

        app = IBApp()
        app.connect_and_start(args.host, args.port, args.client_id)

        all_quotes = {}
        all_trades = {}
        for sym in args.symbols:
            print(f"Fetching {sym} …")
            contract = make_contract(sym, args.currency, args.exchange)
            win = FetchWindow(start_utc=start_epoch, end_utc=end_epoch)
            q = fetch_historical_bidask(app, contract, win, rth_only=args.rth_only, req_timeout=args.req_timeout)
            t = fetch_historical_trades(app, contract, win, rth_only=args.rth_only, req_timeout=args.req_timeout)
            all_quotes[sym] = q
            all_trades[sym] = t
            print(f"  {q.size} quotes, {t.size} trades")

        meta = dict(
            symbols=args.symbols,
            currency=args.currency,
            exchange=args.exchange,
            start_et=args.start_et,
            end_et=args.end_et,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            rth_only=args.rth_only,
            input_tz="US/Eastern",
            ib_tz="UTC"
        )

        save_args = {}
        if len(args.symbols) > 1:
            for sym in args.symbols:
                save_args[f"quotes_{sym}"] = all_quotes[sym]
                save_args[f"trades_{sym}"] = all_trades[sym]
        else:
            sym = args.symbols[0]
            save_args[f"quotes_{sym}"] = all_quotes[sym]
            save_args[f"trades_{sym}"] = all_trades[sym]
            del meta["symbols"]

        save_args["meta"] = np.array([meta], dtype=object)
        np.savez_compressed(args.out, **save_args)
        print(f"Saved → {args.out}")
        app.disconnect()

    # ------------------------------------------------------------------
    elif args.cmd == "sample":
        all_quotes, all_trades, meta = load_npz(args.infile)
        symbols = meta["symbols"]
        selected = args.symbols if args.symbols else symbols

        global_t0 = np.inf
        global_t1 = -np.inf
        for sym in selected:
            q = all_quotes[sym]
            t = all_trades[sym]
            if q.size:
                global_t0 = min(global_t0, int(q["time"][0]))
                global_t1 = max(global_t1, int(q["time"][-1]))
            if t.size:
                global_t1 = max(global_t1, int(t["time"][-1]))
        if global_t0 == np.inf:
            raise ValueError("No data for selected symbols.")
        edges = make_bins(global_t0, global_t1 + args.dt_sec, args.dt_sec)

        save_args = {}
        for sym in selected:
            print(f"Sampling {sym} …")
            sampled, info = sample_to_bins(all_quotes[sym], all_trades[sym], args.dt_sec, edges=edges)
            if len(selected) > 1:
                save_args[f"sampled_{sym}"] = sampled
                save_args[f"info_{sym}"] = np.array([info], dtype=object)
            else:
                save_args["sampled"] = sampled
                save_args["info"] = np.array([info], dtype=object)
        save_args["meta"] = np.array([meta], dtype=object)
        np.savez_compressed(args.outfile, **save_args)
        print(f"Sampled {len(edges)-1} bins → {args.outfile}")

    # ------------------------------------------------------------------
    elif args.cmd == "update":
        app = IBApp()
        app.connect_and_start(args.host, args.port, args.client_id)

        update_file_with_missing_symbols(
            infile=args.infile,
            app=app,
            add_symbols=[s.upper() for s in args.symbols],
            rth_only=args.rth_only,
            currency=args.currency,
            exchange=args.exchange,
            req_timeout=args.req_timeout,
        )
        app.disconnect()

    # ------------------------------------------------------------------
    try:
        app.disconnect()
    except Exception:
        pass
