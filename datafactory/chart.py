from __future__ import annotations

"""
IB Chart Viewer — queued range runs + console
--------------------------------------------

Requirements (install first):
    pip install ib_insync pyqtgraph PyQt5 python-dateutil

Notes:
- Connects to a running TWS or IB Gateway on localhost:7497 (paper) by default.
- Symbol: simple stock ticker (e.g., AAPL). You can adapt the contract creation as needed.
- Start Date:
    • If Duration is set to "Auto (from start)", the app computes the duration from Start Date → now.
    • Otherwise, the selected Duration overrides the Start Date.
- Bar Size: standard IB API bar sizes.
- Duration: common IB API duration strings. Make sure the selected duration is valid for the chosen bar size per IB rules.
- Mouse interaction: pyqtgraph provides mouse-wheel zoom and panning out of the box. Drag with left mouse to pan, wheel to zoom.
- Y-Autozoom checkbox and full-chart crosshair (vertical + horizontal lines following the cursor).
- Interaction combobox with two modes — "Pan/Zoom" and "Select Range". In "Select Range" mode, click once to set start, click
  again to set end.
- NEW (Oct 2025): After the second click we **enqueue** a job instead of launching immediately. Jobs are drained sequentially via
  QProcess. A docked Console shows the full stdout/stderr stream and completion lines. Status bar has Queue: counter, Cancel current,
  and Flush queue buttons.

Tested with: Python 3.10+, ib_insync 0.9+, pyqtgraph 0.13+, PyQt5 5.15+
"""

import sys
import os
import math
from datetime import datetime, date, timezone
from collections import deque
from dataclasses import dataclass

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph import SignalProxy
from ib_insync import IB, Stock, Forex, Contract, util
from zoneinfo import ZoneInfo


# -------------------------
# Candlestick graphics item
# -------------------------
class CandlestickItem(pg.GraphicsObject):
    """Simple and fast candlestick item for pyqtgraph.

    Expects data as a list of dicts with keys:
        time (float seconds since epoch), open, high, low, close
    """
    def __init__(self, data: list[dict]):
        super().__init__()
        self._data = data
        self._picture = None
        self.setFlag(self.ItemUsesExtendedStyleOption)

    def setData(self, data: list[dict]):
        self._data = data
        self._picture = None
        self.prepareGeometryChange()
        self.update()

    def data(self):
        return self._data

    def boundingRect(self):
        if not self._data:
            return QtCore.QRectF()
        xs = [d['time'] for d in self._data]
        lows = [d['low'] for d in self._data]
        highs = [d['high'] for d in self._data]
        return QtCore.QRectF(min(xs), min(lows), max(xs) - min(xs) or 1, max(highs) - min(lows) or 1)

    def paint(self, p: QtGui.QPainter, *_):
        if self._picture is None:
            self._picture = QtGui.QPicture()
            p2 = QtGui.QPainter(self._picture)
            p2.setRenderHint(QtGui.QPainter.Antialiasing, False)

            w = self._bar_width()
            pen_up = QtGui.QPen(QtCore.Qt.NoPen)
            pen_dn = QtGui.QPen(QtCore.Qt.NoPen)
            brush_up = QtGui.QBrush(QtGui.QColor(0, 130, 80))
            brush_dn = QtGui.QBrush(QtGui.QColor(170, 50, 50))
            pen_wick = QtGui.QPen(QtGui.QColor(200, 200, 200))
            pen_wick.setWidth(1)
            pen_wick.setCosmetic(True)

            for d in self._data:
                x = d['time']
                o, h, l, c = d['open'], d['high'], d['low'], d['close']
                # wick
                p2.setPen(pen_wick)
                p2.drawLine(QtCore.QPointF(x, l), QtCore.QPointF(x, h))
                # body
                up = c >= o
                brush = brush_up if up else brush_dn
                pen = pen_up if up else pen_dn
                p2.setPen(pen)
                p2.setBrush(brush)
                y1, y2 = (o, c) if up else (c, o)
                rect = QtCore.QRectF(x - w/2.0, y1, w, max(y2 - y1, 0.0001))
                p2.drawRect(rect)

            p2.end()
        p.drawPicture(0, 0, self._picture)

    def _bar_width(self) -> float:
        if len(self._data) < 2:
            return 60.0  # 1 min default width in x-axis seconds
        # Use 80% of the median spacing
        gaps = [self._data[i+1]['time'] - self._data[i]['time'] for i in range(len(self._data)-1)]
        gaps = sorted(g for g in gaps if g > 0)
        if not gaps:
            return 60.0
        med = gaps[len(gaps)//2]
        return med * 0.8


# -------------------------
# Main Application Window
# -------------------------
@dataclass
class FetchJob:
    symbol: str
    start_et: str
    end_et: str
    out_path: str
    x1: float
    x2: float


def _ts():
    return datetime.now().strftime('%H:%M:%S')


class ChartViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IB Chart Viewer — PyQtGraph")
        self.resize(1200, 900)

        # Central widget & layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Top controls
        controls = QtWidgets.QHBoxLayout()
        vbox.addLayout(controls)

        self.symbolEdit = QtWidgets.QLineEdit()
        self.symbolEdit.setPlaceholderText("Symbol, e.g. AG")
        self.symbolEdit.setText("AG")

        self.startDateEdit = QtWidgets.QDateEdit(calendarPopup=True)
        self.startDateEdit.setDisplayFormat("yyyy-MM-dd")
        self.startDateEdit.setDate(QtCore.QDate.currentDate().addMonths(-1))

        self.barSizeCombo = QtWidgets.QComboBox()
        self.barSizeCombo.addItems([
            "1 secs", "5 secs", "10 secs", "15 secs", "30 secs",
            "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins", "20 mins", "30 mins",
            "1 hour", "2 hours", "3 hours", "4 hours", "8 hours",
            "1 day", "1 week", "1 month"
        ])
        self.barSizeCombo.setCurrentText("5 mins")

        self.durationCombo = QtWidgets.QComboBox()
        self.durationCombo.addItems([
            "Auto (from start)",
            "1 D", "2 D", "3 D", "5 D",
            "1 W", "2 W", "3 W", "4 W",
            "1 M", "2 M", "3 M", "6 M",
            "1 Y", "2 Y", "5 Y", "10 Y"
        ])
        self.durationCombo.setCurrentText("Auto (from start)")

        # SecType selection
        self.secTypeCombo = QtWidgets.QComboBox()
        self.secTypeCombo.addItems([
            "Auto",
            "Stock",
            "Forex (CASH)",
            "Commodity (CMDTY)",
            "CFD",
        ])
        self.secTypeCombo.setCurrentText("Auto")

        # Exchange selection
        self.exchangeCombo = QtWidgets.QComboBox()
        self.exchangeCombo.addItems([
            "Auto",
            "SMART",
            "IDEALPRO",
        ])
        self.exchangeCombo.setCurrentText("Auto")

        # Prefer Volume
        self.preferVolCheck = QtWidgets.QCheckBox("Prefer Volume")
        self.preferVolCheck.setChecked(True)

        self.autoYCheck = QtWidgets.QCheckBox("Y Autozoom")
        self.autoYCheck.setChecked(True)
        self.autoYCheck.toggled.connect(self.onAutoYToggle)

        # Interaction mode
        self.interactionCombo = QtWidgets.QComboBox()
        self.interactionCombo.addItems(["Pan/Zoom", "Select Range"])
        self.interactionCombo.setCurrentText("Pan/Zoom")
        self.interactionCombo.currentTextChanged.connect(self.onInteractionModeChanged)

        self.loadBtn = QtWidgets.QPushButton("Load")
        self.loadBtn.clicked.connect(self.onLoad)

        for lab, w in [
            ("Symbol:", self.symbolEdit),
            ("Start date:", self.startDateEdit),
            ("Bar size:", self.barSizeCombo),
            ("Duration:", self.durationCombo),
            ("SecType:", self.secTypeCombo),
            ("Exchange:", self.exchangeCombo),
        ]:
            lbl = QtWidgets.QLabel(lab)
            lbl.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
            lbl.setMinimumWidth(80)
            controls.addWidget(lbl)
            controls.addWidget(w)

        controls.addWidget(self.autoYCheck)
        controls.addWidget(self.preferVolCheck)
        controls.addWidget(QtWidgets.QLabel("Interaction:"))
        controls.addWidget(self.interactionCombo)
        controls.addWidget(self.loadBtn)
        controls.addStretch(1)

        # Plot area with two vertically stacked plots: candles (top) + volume (bottom)
        pg.setConfigOptions(antialias=False, background="k", foreground="w")
        self.plotWidget = pg.PlotWidget()
        self.volPlotWidget = pg.PlotWidget()
        vbox.addWidget(self.plotWidget, 3)
        vbox.addWidget(self.volPlotWidget, 1)

        # Configure plots
        self.plotWidget.showGrid(x=True, y=True, alpha=0.15)
        self.plotWidget.setMouseEnabled(x=True, y=True)
        self.plotWidget.setMenuEnabled(True)
        self.plotWidget.setClipToView(True)

        self.volPlotWidget.showGrid(x=True, y=True, alpha=0.15)
        self.volPlotWidget.setMouseEnabled(x=True, y=True)
        self.volPlotWidget.setMenuEnabled(True)
        self.volPlotWidget.setClipToView(True)

        # Date axis only on the volume (bottom) plot
        axis_bottom = pg.DateAxisItem(orientation='bottom')
        self.volPlotWidget.setAxisItems({'bottom': axis_bottom})
        self.plotWidget.getPlotItem().hideAxis('bottom')
        # Link X axes
        self.volPlotWidget.setXLink(self.plotWidget)

        # Candle item
        self.candleItem = CandlestickItem([])
        self.plotWidget.addItem(self.candleItem)

        # Volume bars item
        self.volBars = None  # pg.BarGraphItem created on load

        # Crosshair
        self.vLine_main = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((220, 220, 220), width=1, cosmetic=True))
        self.hLine_main = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((220, 220, 220), width=1, cosmetic=True))
        self.vLine_vol = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((220, 220, 220), width=1, cosmetic=True))
        for ln in (self.vLine_main, self.hLine_main, self.vLine_vol):
            ln.setZValue(1000)
        self.plotWidget.addItem(self.vLine_main, ignoreBounds=True)
        self.plotWidget.addItem(self.hLine_main, ignoreBounds=True)
        self.volPlotWidget.addItem(self.vLine_vol, ignoreBounds=True)
        # Mouse move proxies
        self._mouseProxyMain = SignalProxy(self.plotWidget.scene().sigMouseMoved, rateLimit=60, slot=self.onMouseMoved)
        self._mouseProxyVol = SignalProxy(self.volPlotWidget.scene().sigMouseMoved, rateLimit=60, slot=self.onMouseMoved)

        # ViewBoxes & signals
        self.vb = self.plotWidget.getViewBox()
        self.vb_vol = self.volPlotWidget.getViewBox()
        self.vb.sigRangeChanged.connect(self.onRangeChanged)

        # Range selection helpers
        self.rangeRegion = pg.LinearRegionItem(values=[0, 0], orientation=pg.LinearRegionItem.Vertical, movable=False)
        self.rangeRegion.setZValue(999)
        self.rangeRegion.setVisible(False)
        self.plotWidget.addItem(self.rangeRegion)
        self._rangeClicks = []  # store two x-values (float timestamps)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Disconnected — set up TWS/Gateway and click Load")

        # --- Console dock ---
        self.consoleDock = QtWidgets.QDockWidget("Console", self)
        self.consoleDock.setObjectName("ConsoleDock")
        self.consoleDock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
        self.consoleText = QtWidgets.QTextEdit()
        self.consoleText.setReadOnly(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.consoleText.setFont(font)
        self.consoleDock.setWidget(self.consoleText)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.consoleDock)

        # --- Queue controls in status bar ---
        self.queueLabel = QtWidgets.QLabel("Queue: 0")
        self.cancelBtn = QtWidgets.QPushButton("Cancel current")
        self.flushBtn = QtWidgets.QPushButton("Flush queue")
        for b in (self.cancelBtn, self.flushBtn):
            b.setAutoDefault(False)
            b.setDefault(False)
        queueWidget = QtWidgets.QWidget()
        qLayout = QtWidgets.QHBoxLayout(queueWidget)
        qLayout.setContentsMargins(0, 0, 0, 0)
        qLayout.setSpacing(6)
        qLayout.addWidget(self.queueLabel)
        qLayout.addWidget(self.cancelBtn)
        qLayout.addWidget(self.flushBtn)
        self.status.addPermanentWidget(queueWidget)
        self.cancelBtn.clicked.connect(self._cancel_current_job)
        self.flushBtn.clicked.connect(self._flush_queue)

        # IB connection
        self.ib = IB()
        self.connected = False

        # Connect scene click for range selection
        self.plotWidget.scene().sigMouseClicked.connect(self.onSceneClicked)

        # Background process state + queue
        self.proc: QtCore.QProcess | None = None
        self.jobQueue: deque[FetchJob] = deque()
        self._runningJob: FetchJob | None = None
        self._cancelRequested: bool = False
        self._update_queue_ui()

    # ---------------- Contract helpers ----------------
    def _qualify_best_contract(self, symbol: str) -> Contract:
        s = symbol.upper().replace("/", "")
        pref = self.secTypeCombo.currentText()
        ex_pref = self.exchangeCombo.currentText()
        exch = None if ex_pref == "Auto" else ex_pref
        candidates = []
        # Metals heuristics (spot gold/silver)
        if s in ("XAUUSD", "XAGUSD"):
            if pref in ("Auto", "Commodity (CMDTY)"):
                candidates += [
                    Contract(secType='CMDTY', symbol=s, exchange=exch or 'SMART', currency='USD'),
                    Contract(secType='CMDTY', localSymbol=s, exchange=exch or 'SMART'),
                ]
            if pref in ("Auto", "CFD"):
                candidates += [
                    Contract(secType='CFD', symbol=s, exchange=exch or 'SMART', currency='USD'),
                    Contract(secType='CFD', localSymbol=s, exchange=exch or 'SMART'),
                ]

        # Generic FX pair (e.g., EURUSD, GBPUSD, XAUUSD/XAGUSD sometimes as FX)
        if len(s) == 6 and s.isalpha() and pref in ("Auto", "Forex (CASH)"):
            fx = Forex(s)
            if exch:
                fx.exchange = exch
            candidates.append(fx)
        # Stock on SMART (if requested or Auto)
        if pref in ("Auto", "Stock"):
            candidates.append(Stock(symbol, exchange=exch or 'SMART', currency='USD'))
        # Also try CFD generally for any symbol if explicitly requested
        if pref == "CFD":
            candidates.append(Contract(secType='CFD', symbol=symbol, exchange='SMART', currency='USD'))

        last_err = None
        for c in candidates:
            try:
                qc = self.ib.qualifyContracts(c)
                if qc:
                    return qc[0]
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(last_err or f"No matching contract for {symbol}")

    def _whatToShow_for(self, contract: Contract) -> str:
        try:
            st = getattr(contract, 'secType', '').upper()
        except Exception:
            st = ''
        if st in {"CASH", "CFD", "CMDTY", "BOND"}:
            return 'MIDPOINT'
        return 'TRADES'

    def _whatToShow_candidates(self, contract: Contract) -> list:
        # Try options that may carry volume; order of preference depends on secType
        st = getattr(contract, 'secType', '').upper()
        if st in {"CFD", "CMDTY"}:
            return ['TRADES', 'BID', 'ASK', 'MIDPOINT']
        if st == 'CASH':
            return ['TRADES', 'BID', 'ASK', 'MIDPOINT']
        return ['TRADES', 'MIDPOINT']

    def _has_volume(self, bars) -> bool:
        try:
            return any(getattr(b, 'volume', 0) not in (0, None) for b in bars)
        except Exception:
            return False

    def connectIB(self) -> bool:
        if self.connected:
            return True
        try:
            self.ib.connect(host='127.0.0.1', port=7497, clientId=20, readonly=True, timeout=3.0)
            self.connected = True
            self.status.showMessage("Connected to IB @ 127.0.0.1:7497 (clientId=19)")
        except Exception as e:
            self.status.showMessage(f"Failed to connect: {e}")
            self.connected = False
        return self.connected

    def computeDurationFromStart(self, start_qdate: QtCore.QDate) -> str:
        start_dt = datetime(start_qdate.year(), start_qdate.month(), start_qdate.day(), tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta_days = max(1, (now - start_dt).days)
        if delta_days < 14:
            return f"{delta_days} D"
        weeks = delta_days // 7
        if weeks < 8:
            return f"{max(1, weeks)} W"
        months = delta_days // 30
        if months < 24:
            return f"{max(1, months)} M"
        years = delta_days // 365
        return f"{max(1, years)} Y"

    def onLoad(self):
        symbol = self.symbolEdit.text().strip().upper()
        if not symbol:
            self.status.showMessage("Please enter a symbol.")
            return

        bar_size = self.barSizeCombo.currentText()
        duration_choice = self.durationCombo.currentText()
        if duration_choice.startswith("Auto"):
            duration = self.computeDurationFromStart(self.startDateEdit.date())
        else:
            duration = duration_choice

        if not self.connectIB():
            return

        # Build a contract robustly (stocks, FX, metals like XAUUSD/XAGUSD)
        try:
            contract = self._qualify_best_contract(symbol)
        except Exception as e:
            self.status.showMessage(f"Contract error: {e}")
            return

        # Decide data type(s) depending on secType
        wts_list = self._whatToShow_candidates(contract)
        if not self.preferVolCheck.isChecked():
            wts_list = [self._whatToShow_for(contract)]

        # Request historical data (ending now UTC), try multiple whatToShow until we get volume (if preferred)
        end_dt = ''
        bars = None
        last_err = None
        for wts in wts_list:
            try:
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=wts,
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False
                )
                if bars and (not self.preferVolCheck.isChecked() or self._has_volume(bars)):
                    break
            except Exception as e:
                last_err = e
                bars = None
                continue
        if not bars:
            self.status.showMessage(f"No data returned. Last error: {last_err}")
            self.candleItem.setData([])
            if self.volBars:
                self.volPlotWidget.removeItem(self.volBars)
                self.volBars = None
            return

        # Convert bars to candlestick + volume (UTC timestamps on x-axis)
        data = []
        times = []
        vols = []
        for b in bars:
            if isinstance(b.date, str):
                dt = util.parseIBDatetime(b.date)
            elif isinstance(b.date, datetime):
                dt = b.date
            elif isinstance(b.date, date):
                dt = datetime(b.date.year, b.date.month, b.date.day)
            else:
                dt = datetime.utcnow()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = float(dt.timestamp())
            times.append(ts)
            v = float(getattr(b, 'volume', 0.0) or 0.0)
            vols.append(v)
            data.append({'time': ts, 'open': float(b.open), 'high': float(b.high), 'low': float(b.low), 'close': float(b.close), 'volume': v})

        self.candleItem.setData(data)
        self._updateVolume(times, vols)
        self._autoRange()
        self.status.showMessage(f"Loaded {len(data)} bars for {symbol} — {bar_size}, {duration} (whatToShow={wts})")

    def _barWidthSeconds(self) -> float:
        return max(1.0, self.candleItem._bar_width())

    def _updateVolume(self, xs, hs):
        if self.volBars:
            try:
                self.volPlotWidget.removeItem(self.volBars)
            except Exception:
                pass
            self.volBars = None
        w = self._barWidthSeconds()
        self.volBars = pg.BarGraphItem(x=xs, height=hs, width=w, brush=pg.mkBrush(120, 120, 160, 180), pen=pg.mkPen(None))
        self.volPlotWidget.addItem(self.volBars)
        vmax = max(hs) if hs else 1.0
        self.volPlotWidget.setYRange(0, vmax * 1.2 if vmax > 0 else 1.0, padding=0)

    def _autoRange(self):
        br = self.candleItem.boundingRect()
        if br.isNull():
            return
        xpad = br.width() * 0.02
        self.plotWidget.setXRange(br.left() - xpad, br.right() + xpad, padding=0)
        self.volPlotWidget.setXRange(br.left() - xpad, br.right() + xpad, padding=0)
        if self.autoYCheck.isChecked():
            self.updateYAutoRange()
        else:
            ypad = br.height() * 0.05
            ymin, ymax = br.top() - ypad, br.bottom() + ypad
            self.plotWidget.setYRange(min(ymin, ymax), max(ymin, ymax), padding=0)
        if self.volBars:
            try:
                ys = self.volBars.opts.get('height', [])
                vmax = max(ys) if ys else 1.0
                self.volPlotWidget.setYRange(0, vmax * 1.2 if vmax > 0 else 1.0, padding=0)
            except Exception:
                pass

    def onRangeChanged(self, *_):
        if self.autoYCheck.isChecked():
            self.updateYAutoRange()
        if self.volBars:
            x1, x2 = self.vb.viewRange()[0]
            xs = self.volBars.opts['x']
            hs = self.volBars.opts['height']
            vis_h = [h for (x, h) in zip(xs, hs) if x1 <= x <= x2]
            vmax = max(vis_h) if vis_h else (max(hs) if hs else 1.0)
            self.volPlotWidget.setYRange(0, vmax * 1.2 if vmax > 0 else 1.0, padding=0)

    def onAutoYToggle(self, checked: bool):
        if checked:
            self.updateYAutoRange()

    def updateYAutoRange(self):
        data = self.candleItem.data()
        if not data:
            return
        x1, x2 = self.vb.viewRange()[0]
        vis = [d for d in data if x1 <= d['time'] <= x2]
        if not vis:
            return
        lo = min(d['low'] for d in vis)
        hi = max(d['high'] for d in vis)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= 0:
            return
        pad = (hi - lo) * 0.05 or max(1e-6, hi * 0.01)
        ymin, ymax = lo - pad, hi + pad
        self.plotWidget.setYRange(ymin, ymax, padding=0)

    # ------------- Crosshair -------------
    def onMouseMoved(self, evt):
        pos = evt[0]
        if self.plotWidget.sceneBoundingRect().contains(pos):
            vb = self.vb
            hline_target = self.hLine_main
        elif self.volPlotWidget.sceneBoundingRect().contains(pos):
            vb = self.vb_vol
            hline_target = None
        else:
            return
        mp = vb.mapSceneToView(pos)
        x = mp.x()
        y = mp.y()
        self.vLine_main.setPos(x)
        self.vLine_vol.setPos(x)
        if hline_target is not None:
            hline_target.setPos(y)
        self.status.showMessage(f"t={x:.0f}  y={y:.4f}")

    # ------------- Interaction mode / Range selection -------------
    def onInteractionModeChanged(self, mode: str):
        is_range = (mode == "Select Range")
        self._rangeClicks.clear()
        self.rangeRegion.setVisible(False)
        if is_range:
            self.status.showMessage("Range mode: click start in the top chart, then click end to queue …")
        else:
            self.status.showMessage("Pan/Zoom mode: drag to pan, wheel to zoom.")

    def onSceneClicked(self, mouseEvent):
        if self.interactionCombo.currentText() != "Select Range":
            return
        pos = mouseEvent.scenePos()
        if not self.plotWidget.sceneBoundingRect().contains(pos):
            return
        mp = self.vb.mapSceneToView(pos)
        x = float(mp.x())
        br = self.candleItem.boundingRect()
        if not br.isNull():
            x = max(br.left(), min(br.right(), x))
        self._rangeClicks.append(x)
        if len(self._rangeClicks) == 1:
            self.rangeRegion.setRegion((x, x))
            self.rangeRegion.setVisible(True)
            self.status.showMessage("Start set. Click end in the top chart…")
        elif len(self._rangeClicks) >= 2:
            x1, x2 = self._rangeClicks[0], self._rangeClicks[1]
            if x2 < x1:
                x1, x2 = x2, x1
            self.rangeRegion.setRegion((x1, x2))
            self._enqueue_job(x1, x2)
            self._rangeClicks.clear()

    # ---------------- Queue + process management ----------------
    def _fmt_et(self, ts_utc: float) -> str:
        dt_utc = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
        dt_et = dt_utc.astimezone(ZoneInfo("US/Eastern"))
        return dt_et.strftime("%d.%m.%Y %H:%M:%S")

    def _et_stamp_for_filename(self) -> str:
        now_et = datetime.now(ZoneInfo("US/Eastern"))
        return now_et.strftime("%Y%m%d_%H%M%S")

    def _invoke_datafactory(self, x1: float, x2: float):
        # Backwards-compat shim: now just enqueue
        self._enqueue_job(x1, x2)

    def _enqueue_job(self, x1: float, x2: float):
        symbol = (self.symbolEdit.text().strip().upper() or "SYMB")
        start_et = self._fmt_et(x1)
        end_et = self._fmt_et(x2)
        et_stamp = self._et_stamp_for_filename()
        out_path = f"data/{symbol}_ticks_{et_stamp}_rth.npz"
        job = FetchJob(symbol, start_et, end_et, out_path, x1, x2)
        self.jobQueue.append(job)
        self._log(f"Queued: {symbol}  {start_et} → {end_et}  (out: {out_path})", also_status=True)
        self._update_queue_ui()
        if self.proc is None:
            self._start_next_job()

    def _start_next_job(self):
        if self.proc is not None or not self.jobQueue:
            self._update_queue_ui()
            return
        job = self.jobQueue.popleft()
        self._runningJob = job
        self._cancelRequested = False
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'datafactory.py'),
            'fetch',
            '--symbols', job.symbol,
            "XAGUSD",
            "XAUUSD",
            #'--exchange', 'SMART',
            #'--currency', 'USD',
            '--start-et', job.start_et,
            '--end-et', job.end_et,
            '--out', job.out_path,
        ]
        shown = ' '.join([
            'python', 'datafactory.py', 'fetch',
            '--symbols', job.symbol,
            "XAGUSD",
            "XAUUSD",
            #'--exchange', 'SMART',
            #'--currency', 'USD',
            '--start-et', f'"{job.start_et}"',
            '--end-et', f'"{job.end_et}"',
            '--out', job.out_path,
        ])
        self._log(f"Starting: {cmd}", also_status=True)

        self.proc = QtCore.QProcess(self)
        self.proc.setWorkingDirectory(os.path.dirname(__file__))
        self.proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(self.proc.readAllStandardOutput()))
        self.proc.readyReadStandardError.connect(lambda: self._append_proc_output(self.proc.readAllStandardError()))
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.start(cmd[0], cmd[1:])
        self._update_queue_ui()

    def _on_proc_finished(self, code: int, status: QtCore.QProcess.ExitStatus):
        job = self._runningJob
        self._runningJob = None
        cancelled = self._cancelRequested
        self._cancelRequested = False
        sym = job.symbol if job else "?"
        rng = f"{job.start_et} → {job.end_et}" if job else "?"
        if cancelled:
            self._log(f"Cancelled: {sym}  {rng}", also_status=True)
        else:
            self._log(f"Done: {sym}  {rng}  → exit code {code}", also_status=True)
        self.proc = None
        self._update_queue_ui()
        self._start_next_job()

    def _append_proc_output(self, qbytearray):
        try:
            text = bytes(qbytearray).decode('utf-8', errors='replace')
        except Exception:
            text = str(qbytearray)
        if text:
            self.consoleText.append(text.rstrip("\n"))
            self.consoleText.moveCursor(QtGui.QTextCursor.End)
            last = [ln for ln in text.splitlines() if ln.strip()]
            if last:
                self.status.showMessage(last[-1])
        self._update_queue_ui()

    def _cancel_current_job(self):
        if self.proc is None:
            self._log("No running job to cancel.")
            return
        self._cancelRequested = True
        self._log("Cancelling current job…", also_status=True)
        self.proc.terminate()
        QtCore.QTimer.singleShot(3000, lambda: self.proc and self.proc.state() != QtCore.QProcess.NotRunning and self.proc.kill())

    def _flush_queue(self):
        n = len(self.jobQueue)
        self.jobQueue.clear()
        self._log(f"Flushed {n} queued job(s).", also_status=True)
        self._update_queue_ui()

    def _update_queue_ui(self):
        pending = len(self.jobQueue)
        running = self.proc is not None and self.proc.state() != QtCore.QProcess.NotRunning
        self.queueLabel.setText(f"Queue: {pending + (1 if running else 0)}")
        self.cancelBtn.setEnabled(running)
        self.flushBtn.setEnabled(pending > 0 or running)

    # ------------- Logging helper -------------
    def _log(self, msg: str, also_status: bool = False):
        self.consoleText.append(f"[{_ts()}] {msg}")
        self.consoleText.moveCursor(QtGui.QTextCursor.End)
        if also_status:
            self.status.showMessage(msg)

    # ---------------- Utilities ----------------
    def _fmt_cli_range(self, start_et: str, end_et: str) -> str:
        return f'"{start_et}" → "{end_et}"'


# -------------------------
# Entrypoint
# -------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ChartViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
