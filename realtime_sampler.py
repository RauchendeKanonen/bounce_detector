import time
import math
import threading
from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


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


def _now_sec() -> int:
    # Wall clock in integer seconds (UTC-like). Use time.time() source.
    return int(time.time())


class RealtimeSampler:
    """
    Real-time sampler for best bid/ask + trades at fixed Δt.
    - Call push_quote(...) and push_trade(...) from your market data callbacks.
    - Start the sampler: start()
    - Fetch a window (thread-safe): get_recent_window(n_bins, fields, as_torch)

    Sampling logic matches the offline version:
    - Quotes: snapshot = last in bin
    - Sizes: last in bin
    - Last trade in bin, trade volume (sum), buy/sell counts (aggressor vs prevailing BBO),
      updates_count (quote updates per bin)
    - Derived: mid, spread, microprice, ret_1 (log mid diff), OFI-like deltas of bests, flow_intensity
    """

    def __init__(
        self,
        dt_sec: int,
        buffer_bins: int,
        align_to: Optional[int] = None,
        include_trades: bool = True,
        auto_persist_path: Optional[str] = None,
    ):
        """
        dt_sec: bin size in seconds
        buffer_bins: ring buffer length (how many recent bins to keep)
        align_to: epoch seconds to align the bin grid; if None uses current second
        include_trades: whether to process trades
        auto_persist_path: if set, sampler will periodically save buffer snapshot (not on every bin)
        """
        assert dt_sec >= 1
        assert buffer_bins >= 2

        self.dt = int(dt_sec)
        self.include_trades = include_trades
        self.auto_persist_path = auto_persist_path

        # Input queues (append-only) guarded by lock
        self._inq_lock = threading.Lock()
        self._quotes = deque()  # items: (ts:int, bid:float, ask:float, bs:int, asz:int)
        self._trades = deque()  # items: (ts:int, price:float, size:int)

        # Output ring buffer
        self._buf_lock = threading.Lock()
        self._buf = np.zeros(buffer_bins, dtype=SAMPLED_DTYPE)
        self._buf.fill(np.nan)
        # integer fields must be valid ints; we’ll zero them separately
        for k in ("buy_count", "sell_count", "updates_count"):
            self._buf[k] = 0
        self._N = buffer_bins
        self._widx = 0  # next write index
        self._bins_written = 0  # grows unbounded, for global indexing

        # State for deltas/returns
        self._prev_bid = np.nan
        self._prev_ask = np.nan
        self._prev_bid_size = np.nan
        self._prev_ask_size = np.nan
        self._prev_mid = np.nan

        # Bin clock
        t0 = _now_sec() if align_to is None else int(align_to)
        self._cur_bin_left = (t0 // self.dt) * self.dt
        self._cur_bin_right = self._cur_bin_left + self.dt

        # Thread lifecycle
        self._thr = None
        self._stop_evt = threading.Event()

        # Optional persistence pacing
        self._last_persist_ts = 0
        self._persist_min_gap = 2.0  # seconds

    # ---------- Public thread-safe inputs ----------

    def push_quote(self, ts_sec: int, bid: float, ask: float, bid_size: int, ask_size: int):
        """Push a top-of-book update. ts_sec must be integer seconds. Thread-safe."""
        with self._inq_lock:
            self._quotes.append((int(ts_sec), float(bid), float(ask), int(bid_size), int(ask_size)))

    def push_trade(self, ts_sec: int, price: float, size: int):
        """Push a last trade. ts_sec must be integer seconds. Thread-safe."""
        if not self.include_trades:
            return
        with self._inq_lock:
            self._trades.append((int(ts_sec), float(price), int(size)))

    # ---------- Public thread-safe outputs ----------

    def get_recent_window(
        self,
        n_bins: int,
        fields: Optional[Sequence[str]] = None,
        as_torch: bool = False,
        return_times: bool = False,
    ):
        """
        Get the most recent n_bins from the ring buffer (oldest→newest order).
        - fields: subset of columns; if None returns the full structured array
        - as_torch: return torch.tensor [n_bins, len(fields)] (requires torch)
        - return_times: if True, also return (t_left, t_right) arrays separately for convenience
        """
        with self._buf_lock:
            n = min(n_bins, min(self._bins_written, self._N))
            if n <= 0:
                if as_torch:
                    if not _HAS_TORCH:
                        raise RuntimeError("Torch not available")
                    return torch.empty((0, 0)), None if return_times else torch.empty((0, 0))
                return np.empty(0, dtype=self._buf.dtype)

            # Compute slice across ring buffer
            end = (self._widx - 1) % self._N
            start = (end - (n - 1)) % self._N
            if start <= end:
                view = self._buf[start:end + 1].copy()
            else:
                view = np.concatenate([self._buf[start:], self._buf[:end + 1]], axis=0).copy()

        # Field selection
        if fields:
            mat = np.column_stack([view[f] for f in fields])
        else:
            mat = view

        if as_torch:
            if not _HAS_TORCH:
                raise RuntimeError("Torch not available; install torch or use as_torch=False")
            tens = torch.from_numpy(np.asarray(mat))
            if return_times:
                return tens, (view["t_left"].copy(), view["t_right"].copy())
            return tens

        if return_times:
            return mat, (view["t_left"].copy(), view["t_right"].copy())
        return mat

    # Optional snapshot to disk (same format)
    def save_npz(self, path: str):
        with self._buf_lock:
            # Export only valid bins in chronological order
            n = min(self._bins_written, self._N)
            if n <= 0:
                arr = self._buf[:0].copy()
            else:
                end = (self._widx - 1) % self._N
                start = (end - (n - 1)) % self._N
                if start <= end:
                    arr = self._buf[start:end + 1].copy()
                else:
                    arr = np.concatenate([self._buf[start:], self._buf[:end + 1]], axis=0).copy()
        meta = dict(ts=time.time(), dt_sec=self.dt, n_bins=int(arr.shape[0]), live=True)
        np.savez_compressed(path, sampled=arr, info=np.array([meta], dtype=object))

    # ---------- Thread lifecycle ----------

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop_evt.clear()
        self._thr = threading.Thread(target=self._run_loop, daemon=True)
        self._thr.start()

    def stop(self, join: bool = True, timeout: Optional[float] = 2.0):
        self._stop_evt.set()
        if join and self._thr:
            self._thr.join(timeout=timeout)

    # ---------- Internal: sampling loop ----------

    def _run_loop(self):
        # We run a “tick” roughly every 10–20 ms to check if a bin closed.
        poll = 0.02
        while not self._stop_evt.is_set():
            now = _now_sec()
            # Process any closed bins up to 'now'
            while self._cur_bin_right <= now:
                q_list, t_list = self._drain_inputs_until(self._cur_bin_right - 1)
                self._commit_bin(self._cur_bin_left, self._cur_bin_right, q_list, t_list)
                self._cur_bin_left = self._cur_bin_right
                self._cur_bin_right = self._cur_bin_left + self.dt

            time.sleep(poll)

            # Opportunistic persistence
            if self.auto_persist_path:
                ts = time.time()
                if ts - self._last_persist_ts >= self._persist_min_gap:
                    try:
                        self.save_npz(self.auto_persist_path)
                        self._last_persist_ts = ts
                    except Exception:
                        pass  # best-effort

    # Drain inputs up to and including cutoff second
    def _drain_inputs_until(self, cutoff_sec: int) -> Tuple[list, list]:
        with self._inq_lock:
            q_out = []
            while self._quotes and self._quotes[0][0] <= cutoff_sec:
                q_out.append(self._quotes.popleft())
            t_out = []
            if self.include_trades:
                while self._trades and self._trades[0][0] <= cutoff_sec:
                    t_out.append(self._trades.popleft())
        return q_out, t_out

    def _commit_bin(self, t_left: int, t_right: int, q_list: list, t_list: list):
        """
        Build one bin from drained ticks and write to ring buffer.
        """
        # Quote snapshot: take the last quote by timestamp, keep update count
        updates_count = len(q_list)
        bid = self._prev_bid
        ask = self._prev_ask
        bid_size = self._prev_bid_size
        ask_size = self._prev_ask_size

        if updates_count:
            # stable 'last in bin' behavior relies on original order; we drained in FIFO arrival
            last_q = q_list[-1]
            _, bid, ask, bid_size, ask_size = last_q

        # Trades aggregation
        last_trade_price = np.nan
        trade_volume = 0.0
        buy_count = 0
        sell_count = 0

        if self.include_trades and t_list:
            # Last trade (latest timestamp wins; we drained FIFO, so last item is fine)
            last_trade_price = t_list[-1][1]
            # Sum sizes
            trade_volume = float(sum(x[2] for x in t_list))

            # Aggressor classification vs prevailing BBO at each trade time:
            # We approximate using the current bin's prevailing BBO (snapshot) if no micro-BBO stream at each trade.
            # Better: walk quotes with two pointers; here we use end-of-bin snapshot as a proxy.
            pbid = bid
            pask = ask
            if not np.isnan(pbid) and not np.isnan(pask):
                for (_, price, _sz) in t_list:
                    if price >= pask:
                        buy_count += 1
                    elif price <= pbid:
                        sell_count += 1
                    # else ambiguous → ignored

        # Derived features
        mid = (bid + ask) * 0.5 if (not np.isnan(bid) and not np.isnan(ask)) else np.nan
        spread = (ask - bid) if (not np.isnan(bid) and not np.isnan(ask)) else np.nan
        microprice = np.nan
        denom = (bid_size + ask_size) if (not np.isnan(bid_size) and not np.isnan(ask_size)) else np.nan
        if denom and denom != 0 and not np.isnan(denom) and not np.isnan(bid) and not np.isnan(ask):
            microprice = (ask * bid_size + bid * ask_size) / denom

        # Deltas and returns vs previous bin snapshot
        d_bid = np.nan if np.isnan(self._prev_bid) or np.isnan(bid) else (bid - self._prev_bid)
        d_ask = np.nan if np.isnan(self._prev_ask) or np.isnan(ask) else (ask - self._prev_ask)
        d_bid_size = np.nan if np.isnan(self._prev_bid_size) or np.isnan(bid_size) else (bid_size - self._prev_bid_size)
        d_ask_size = np.nan if np.isnan(self._prev_ask_size) or np.isnan(ask_size) else (ask_size - self._prev_ask_size)
        ret_1 = np.nan
        if not np.isnan(mid) and not np.isnan(self._prev_mid) and self._prev_mid > 0 and mid > 0:
            ret_1 = math.log(mid) - math.log(self._prev_mid)

        flow_intensity = float(buy_count - sell_count)

        row = np.zeros((), dtype=SAMPLED_DTYPE)
        row["t_left"] = int(t_left)
        row["t_right"] = int(t_right)
        row["bid"] = float(bid) if not np.isnan(bid) else np.nan
        row["ask"] = float(ask) if not np.isnan(ask) else np.nan
        row["bid_size"] = float(bid_size) if not np.isnan(bid_size) else np.nan
        row["ask_size"] = float(ask_size) if not np.isnan(ask_size) else np.nan
        row["last_trade_price"] = float(last_trade_price) if not np.isnan(last_trade_price) else np.nan
        row["trade_volume"] = float(trade_volume)
        row["buy_count"] = int(buy_count)
        row["sell_count"] = int(sell_count)
        row["updates_count"] = int(updates_count)
        row["mid"] = float(mid) if not np.isnan(mid) else np.nan
        row["spread"] = float(spread) if not np.isnan(spread) else np.nan
        row["microprice"] = float(microprice) if not np.isnan(microprice) else np.nan
        row["ret_1"] = float(ret_1) if not np.isnan(ret_1) else np.nan
        row["d_bid"] = float(d_bid) if not np.isnan(d_bid) else np.nan
        row["d_ask"] = float(d_ask) if not np.isnan(d_ask) else np.nan
        row["d_bid_size"] = float(d_bid_size) if not np.isnan(d_bid_size) else np.nan
        row["d_ask_size"] = float(d_ask_size) if not np.isnan(d_ask_size) else np.nan
        row["flow_intensity"] = float(flow_intensity)

        # Commit to ring
        with self._buf_lock:
            self._buf[self._widx] = row
            self._widx = (self._widx + 1) % self._N
            self._bins_written += 1

        # Update prev snapshot for next bin
        self._prev_bid = bid
        self._prev_ask = ask
        self._prev_bid_size = bid_size
        self._prev_ask_size = ask_size
        self._prev_mid = mid
