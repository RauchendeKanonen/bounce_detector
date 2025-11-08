# opt_term_structure.py
import sys
import argparse
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.common import TickerId, TickAttrib
from ibapi.tag_value import TagValue

# PyQt5 / PyQtGraph
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ---------- Utilities ----------

def parse_yyyymmdd(s: str) -> datetime:
    """Parse IB expiration strings like '20251115' or '2025XXXX' (some venues) safely."""
    s = s.strip()
    # Normalize YYYYMM or YYYYMMDD into a concrete date (assume third Friday for month-level)
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, "%Y%m%d")
    if len(s) == 6 and s.isdigit():
        # Approximate: use 3rd Friday of month
        year = int(s[:4])
        month = int(s[4:6])
        # Find third Friday
        # Start at 1st of month
        d = datetime(year, month, 1)
        # weekday(): Monday=0 ... Sunday=6; we want Friday=4
        days_to_first_friday = (4 - d.weekday()) % 7
        first_friday = d.replace(day=1 + days_to_first_friday)
        third_friday_day = first_friday.day + 14
        return d.replace(day=third_friday_day)
    # Fallback: try ISO parsing
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # as last resort, now
        return datetime.now()

def within_expiry_bounds(exp: str, min_exp: Optional[str], max_exp: Optional[str]) -> bool:
    def norm(x):
        if x is None:
            return None
        # allow YYYYMM or YYYYMMDD
        if len(x) == 6:
            x += "01"
        return int(x)
    e = norm(exp if len(exp) in (6,8) else exp[:8])
    mn = norm(min_exp)
    mx = norm(max_exp)
    if mn and e < mn:
        return False
    if mx and e > mx:
        return False
    return True

# ---------- IB App ----------

class IBOptionTermApp(EWrapper, EClient):
    def __init__(self, ui_callback, args):
        EClient.__init__(self, self)
        self.ui_callback = ui_callback
        self.args = args

        self.nextOrderId: Optional[int] = None
        self.connected_event = threading.Event()

        # Underlying resolution
        self.underlying_conid: Optional[int] = None
        self._contract_details_req_id: Optional[int] = None

        # SecDef option params
        self._secdef_req_id: Optional[int] = None
        self.available_expirations: List[str] = []
        self.available_strikes: List[float] = []

        # Market data tracking: map tickerId -> expiry
        self.ticker_to_exp: Dict[int, str] = {}
        # Latest quotes per expiry
        self.quotes: Dict[str, Dict[str, Optional[float]]] = {}
        # To avoid flooding, throttle UI updates
        self._last_ui_push = 0

    # ---- Connection / lifecycle ----
    def connect_and_run(self, host, port, clientId):
        self.connect(host, port, clientId)
        thread = threading.Thread(target=self.run, name="IBApiThread", daemon=True)
        thread.start()
        # give it a moment to connect
        if not self.connected_event.wait(timeout=5.0):
            print("Warning: IB connection not confirmed yet...")

    def nextValidId(self, orderId: int):
        self.nextOrderId = orderId
        self.connected_event.set()
        # kickoff: resolve underlying conId
        self.request_underlying_conid()

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson=None):
        # Log to console and optionally to UI banner
        msg = f"IB ERROR reqId={reqId} code={errorCode} msg={errorString}"
        print(msg)
        # Let UI know about severe issues
        if errorCode in (200, 201, 202, 10182, 354):  # common contract/md errors
            self.ui_callback("status", msg)

    # ---- Underlying resolution ----
    def request_underlying_conid(self):
        req_id = 9001
        self._contract_details_req_id = req_id
        c = Contract()
        c.symbol = self.args.symbol
        c.secType = "STK"
        c.currency = self.args.currency
        c.exchange = self.args.exchange or "SMART"
        self.reqContractDetails(req_id, c)

    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        if reqId == self._contract_details_req_id:
            if contractDetails and contractDetails.contract:
                # Prefer primary listing if multiple; pick first for simplicity
                self.underlying_conid = contractDetails.contract.conId

    def contractDetailsEnd(self, reqId: int):
        if reqId == self._contract_details_req_id:
            if not self.underlying_conid:
                self.ui_callback("status", "Failed to resolve underlying conId.")
                return
            # Request option chain (expiries, strikes)
            self._request_secdef_params()

    # ---- SecDef option parameters ----
    def _request_secdef_params(self):
        req_id = 9101
        self._secdef_req_id = req_id
        # NOTE: include req_id as first argument
        self.reqSecDefOptParams(
			req_id,
			self.args.symbol,
			"",           # futFopExchange
			"STK",        # underlyingSecType
            self.underlying_conid
        )

    def securityDefinitionOptionParameter(self, reqId:int, exchange:str, underlyingConId:int,
                                         tradingClass:str, multiplier:str,
                                         expirations:set, strikes:set):
        if reqId != self._secdef_req_id:
            return
        # Filter to desired exchange if user specified non-SMART; otherwise accept SMART-capable venues
        want_exch = (self.args.exchange or "SMART").upper()
        # Collect all expirations/strikes that match our desired strike & exchange universe
        # Even if exchange is not SMART, we still can route to SMART; keep expirations regardless.
        self.available_expirations = sorted(list(expirations))
        self.available_strikes = sorted([float(s) for s in strikes if self._is_number(s)])

    def securityDefinitionOptionParameterEnd(self, reqId:int):
        if reqId != self._secdef_req_id:
            return
        # Filter expirations and ensure our strike is available if possible
        strike = float(self.args.strike)
        # If strike not in list, we'll still try (IB often accepts non-listed due to rounding)
        expiries = [e for e in self.available_expirations
                    if within_expiry_bounds(e, self.args.min_exp, self.args.max_exp)]
        # Sort by actual date
        expiries.sort(key=lambda e: parse_yyyymmdd(e))
        if self.args.max_expiries and self.args.max_expiries > 0:
            expiries = expiries[: self.args.max_expiries]

        if not expiries:
            self.ui_callback("status", "No expirations matched filters.")
            return

        # Kick off market data requests
        base_id = 10000
        for i, exp in enumerate(expiries):
            tid = base_id + i
            self._req_option_mktdata(tid, exp, strike)
            self.ticker_to_exp[tid] = exp
            self.quotes.setdefault(exp, {"bid": None, "ask": None, "last": None, "mid": None})

        self.ui_callback("status", f"Subscribing to {len(expiries)} expirations for {self.args.symbol} {self.args.right} {strike:g}")

    def _is_number(self, x):
        try:
            float(x)
            return True
        except Exception:
            return False

    def _req_option_mktdata(self, ticker_id:int, expiry:str, strike:float):
        c = Contract()
        c.symbol = self.args.symbol
        c.secType = "OPT"
        c.exchange = self.args.exchange or "SMART"
        c.currency = self.args.currency
        c.lastTradeDateOrContractMonth = expiry  # 'YYYYMM' or 'YYYYMMDD'
        c.strike = float(strike)
        c.right = self.args.right.upper()
        c.multiplier = str(self.args.multiplier)

        generic = ""  # basic bid/ask/last ticks arrive without generic ticks
        snapshot = False
        regulatorySnaphsot = False
        opts: List[TagValue] = []
        self.reqMktData(ticker_id, c, generic, snapshot, regulatorySnaphsot, opts)

    # ---- Market data handlers ----
    def tickPrice(self, reqId:TickerId, tickType:int, price:float, attrib:TickAttrib):
        exp = self.ticker_to_exp.get(reqId)
        if not exp:
            return
        q = self.quotes.setdefault(exp, {"bid": None, "ask": None, "last": None, "mid": None})
        if tickType == 1:       # BID
            q["bid"] = price
        elif tickType == 2:     # ASK
            q["ask"] = price
        elif tickType == 4:     # LAST
            q["last"] = price

        # compute mid if possible
        if q["bid"] is not None and q["ask"] is not None:
            q["mid"] = (q["bid"] + q["ask"]) / 2.0

        # Throttle UI updates to ~10Hz
        now = time.time()
        if now - self._last_ui_push > 0.1:
            self._last_ui_push = now
            self.ui_callback("quotes", self.quotes.copy())

    def tickSize(self, reqId:TickerId, tickType:int, size:int):
        # Not strictly needed for plotting prices
        pass

    # ---- Cleanup ----
    def stop(self):
        try:
            # Cancel all mkt data
            for tid in list(self.ticker_to_exp.keys()):
                self.cancelMktData(tid)
        except Exception:
            pass
        try:
            self.disconnect()
        except Exception:
            pass


# ---------- Qt / Plotting ----------

class TermStructurePlot(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle(f"Option Term Structure – {args.symbol} {args.right}{args.strike:g}")

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        self.status_label = QtWidgets.QLabel("Connecting to IB…")
        layout.addWidget(self.status_label)

        date_axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': date_axis})
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('bottom', 'Expiration Date')
        self.plot.setLabel('left', 'Price')
        self.plot.addLegend(offset=(10, 10))
        layout.addWidget(self.plot, 1)

        self.curve_bid = self.plot.plot([], [], pen=pg.mkPen(width=2), name="Bid")
        self.curve_ask = self.plot.plot([], [], pen=pg.mkPen(width=2), name="Ask")
        self.curve_mid = self.plot.plot([], [], pen=pg.mkPen(width=2), name="Mid")
        self.curve_last = self.plot.plot([], [], pen=pg.mkPen(width=2), name="Last")

        self.latest_quotes = {}

        # ✅ Corrected line:
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start()
    def update_status(self, text: str):
        self.status_label.setText(text)

    def ingest_quotes(self, quotes: Dict[str, Dict[str, Optional[float]]]):
        self.latest_quotes = quotes

    def refresh_plot(self):
        if not self.latest_quotes:
            return
        # Prepare x (epoch timestamps) and y’s
        exp_sorted = sorted(self.latest_quotes.keys(), key=lambda e: parse_yyyymmdd(e))
        xs = [parse_yyyymmdd(e).timestamp() for e in exp_sorted]

        def series(key):
            arr = []
            for e in exp_sorted:
                v = self.latest_quotes.get(e, {}).get(key)
                arr.append(float(v) if v is not None else float('nan'))
            return arr

        ys_bid = series("bid")
        ys_ask = series("ask")
        ys_mid = series("mid")
        ys_last = series("last")

        self.curve_bid.setData(xs, ys_bid)
        self.curve_ask.setData(xs, ys_ask)
        self.curve_mid.setData(xs, ys_mid)
        self.curve_last.setData(xs, ys_last)

    def closeEvent(self, event):
        super().closeEvent(event)


# ---------- Wiring it together ----------

class Controller:
    def __init__(self, args):
        self.args = args
        self.app_qt = QtWidgets.QApplication(sys.argv)
        self.win = TermStructurePlot(args)

        # IB app
        def ui_callback(kind, payload):
            if kind == "status":
                self.win.update_status(str(payload))
            elif kind == "quotes":
                self.win.ingest_quotes(payload)

        self.ib = IBOptionTermApp(ui_callback, args)

        # Ensure clean shutdown when window closes
        self.app_qt.aboutToQuit.connect(self.shutdown)

    def start(self):
        host = self.args.host
        port = self.args.port if not self.args.paper else 7497
        client_id = self.args.client_id

        self.ib.connect_and_run(host, port, client_id)
        self.win.show()
        rc = self.app_qt.exec_()
        return rc

    def shutdown(self):
        self.ib.stop()


def build_argparser():
    p = argparse.ArgumentParser(description="Plot IB option Bid/Ask/Mid/Last vs expiration date for a given stock & strike.")
    p.add_argument("--symbol", required=True, help="Underlying stock symbol (e.g., AAPL)")
    p.add_argument("--right", required=True, choices=["C","P","c","p"], help="Option right: C or P")
    p.add_argument("--strike", required=True, type=float, help="Option strike price")

    p.add_argument("--exchange", default="SMART", help="Exchange to route (default SMART)")
    p.add_argument("--currency", default="USD", help="Currency (default USD)")
    p.add_argument("--multiplier", default=100, type=int, help="Contract multiplier (default 100)")

    p.add_argument("--host", default="127.0.0.1", help="IB Gateway/TWS host")
    p.add_argument("--port", type=int, default=7497, help="IB Gateway/TWS port (7497 paper, 7496 live)")
    p.add_argument("--client-id", type=int, default=123, help="IB API client id")
    p.add_argument("--paper", action="store_true", help="Use paper trading port 7497")

    p.add_argument("--max-expiries", type=int, default=10, help="Maximum number of expirations to subscribe")
    p.add_argument("--min-exp", type=str, default=None, help="Minimum expiration filter (YYYYMM or YYYYMMDD)")
    p.add_argument("--max-exp", type=str, default=None, help="Maximum expiration filter (YYYYMM or YYYYMMDD)")
    return p

def main():
    args = build_argparser().parse_args()
    # Normalize right
    args.right = args.right.upper()
    ctl = Controller(args)
    sys.exit(ctl.start())

if __name__ == "__main__":
    main()
