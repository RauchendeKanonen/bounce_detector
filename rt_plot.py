"""
realtime_plot.py — drop-in realtime plotting server using pyqtgraph.

Goals
- Usable from non-Qt code. Spawns its own Qt event loop in a separate process.
- Realtime plotting: points are visible as soon as you push them.
- Multiple series, each can bind to left or right Y axis (different scales).
- X axis uses Unix timestamps; labels render in a chosen timezone (default US/Eastern).
- Minimal, dependency-light: pyqtgraph + a Qt binding (PyQt5/PySide6) + Python ≥ 3.9.

API (from your non-Qt app)

    from realtime_plot import RealtimeGraph

    g = RealtimeGraph(title="Demo", tzname="America/New_York", max_points=10_000)
    g.add_series("temp_degC", axis="left")
    g.add_series("pressure_kPa", axis="right")

    # Push data (t is Unix seconds)
    g.add_point("temp_degC", t, 23.5)
    g.add_point("pressure_kPa", t, 101.2)

    # Optional niceties
    g.set_labels(left="°C", right="kPa")
    g.set_title("Sensors")
    g.set_x_range(seconds=300)    # keep a sliding 5-minute window

    # When done
    g.close()

Notes
- This module attempts to import PyQt5 first, then PySide6.
- On Windows, ensure your script guards top level code with if __name__ == "__main__":
- The plotting process is robust to bursts: commands are queued, GUI flushes at ~60 FPS max.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.queues import Queue
from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional, Tuple, List
import time
import math
import sys

# --- Qt + pyqtgraph imports -------------------------------------------------
QtBinding = None
try:
    from PyQt5 import QtCore, QtWidgets
    QtBinding = "PyQt5"
except Exception:
    try:
        from PySide6 import QtCore, QtWidgets
        QtBinding = "PySide6"
    except Exception as e:
        raise RuntimeError(
            "No Qt binding found. Please install PyQt5 or PySide6."
        ) from e

import pyqtgraph as pg

# --- Time axis with timezone -------------------------------------------------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    # Fallback for older Python with backports.zoneinfo
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "zoneinfo not available. Use Python ≥3.9 or install backports.zoneinfo"
        ) from e

from datetime import datetime, timezone


class TimeAxisTz(pg.AxisItem):
    """Bottom axis that formats Unix timestamps in a given timezone."""

    def __init__(self, tzname: str = "America/New_York", *args, **kwargs):
        super().__init__(orientation="bottom", *args, **kwargs)
        self.tz = ZoneInfo(tzname)
        self._fmt_short = "%H:%M:%S\n%b %d"
        self._fmt_long = "%Y-%m-%d\n%H:%M:%S"

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (Qt signature)
        # Choose coarser/denser formatting based on spacing
        fmt = self._fmt_short if spacing < 24 * 3600 else self._fmt_long
        out = []
        for v in values:
            try:
                dt = datetime.fromtimestamp(float(v), tz=self.tz)
                out.append(dt.strftime(fmt))
            except Exception:
                out.append("")
        return out


# --- Messages passed across processes ---------------------------------------
@dataclass
class MsgAddSeries:
    name: str
    axis: str  # "left" or "right"
    pen: Optional[Tuple[int, int, int]] = None  # RGB tuple


@dataclass
class MsgAddPoint:
    name: str
    t: float  # Unix seconds
    y: float


@dataclass
class MsgSetLabels:
    left: Optional[str]
    right: Optional[str]


@dataclass
class MsgSetTitle:
    title: str


@dataclass
class MsgSetXRange:
    # exactly one of the fields should be provided
    seconds: Optional[float] = None
    xmin: Optional[float] = None
    xmax: Optional[float] = None


@dataclass
class MsgClose:
    pass


# --- Plotting process --------------------------------------------------------
class _PlotProcess(mp.Process):
    def __init__(self, q: Queue, title: str, tzname: str, max_points: int):
        super().__init__(daemon=True)
        self.q = q
        self.title = title
        self.tzname = tzname
        self.max_points = max_points

        # State per series
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.buffers: Dict[str, Tuple[deque, deque]] = {}  # name -> (xbuf, ybuf)
        self.axis_of: Dict[str, str] = {}  # name -> "left"|"right"

    def run(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        axis = TimeAxisTz(self.tzname)
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": axis})
        self.plot_widget.setWindowTitle(self.title)
        self.plot_item = self.plot_widget.getPlotItem()

        # Improve performance
        pg.setConfigOptions(antialias=True, foreground=(220, 220, 220))
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)

        # Right axis via a secondary ViewBox
        self.plot_item.showAxis("right", True)
        self.right_vb = pg.ViewBox()
        self.plot_item.scene().addItem(self.right_vb)
        self.plot_item.getAxis("right").linkToView(self.right_vb)
        self.right_vb.setXLink(self.plot_item.vb)

        def update_views():
            self.right_vb.setGeometry(self.plot_item.vb.sceneBoundingRect())
            self.right_vb.linkedViewChanged(self.plot_item.vb, self.right_vb.XAxis)

        self.plot_item.vb.sigResized.connect(update_views)

        # Labels
        self.plot_item.setLabel("left", "")
        self.plot_item.getAxis("right").setLabel("")
        self.plot_item.setLabel("bottom", "US Eastern Time")

        self.plot_widget.resize(1000, 600)
        self.plot_widget.show()

        # Timer to pump queue and refresh at ~60 Hz
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._pulse)
        self.timer.start(int(1000 / 60))

        # Also ensure we process any pending at start
        self._pulse()

        app.exec()

    # -- queue handling
    def _pulse(self):
        # Drain queue quickly
        for _ in range(200):
            try:
                msg = self.q.get_nowait()
            except Exception:
                break
            self._handle_msg(msg)

        # Update all curves
        for name, curve in self.curves.items():
            xbuf, ybuf = self.buffers[name]
            if len(xbuf) == 0:
                continue
            curve.setData(list(xbuf), list(ybuf))

    def _handle_msg(self, msg):
        if isinstance(msg, MsgAddSeries):
            if msg.name in self.curves:
                return
            # Choose viewbox based on axis
            if msg.axis not in ("left", "right"):
                msg.axis = "left"
            vb = self.plot_item.vb if msg.axis == "left" else self.right_vb
            pen = msg.pen or pg.intColor(len(self.curves))
            curve = pg.PlotCurveItem(pen=pen, name=msg.name)
            vb.addItem(curve)
            self.curves[msg.name] = curve
            self.buffers[msg.name] = (deque(maxlen=self.max_points), deque(maxlen=self.max_points))
            self.axis_of[msg.name] = msg.axis
            # create legend lazily
            if not hasattr(self, "legend"):
                self.legend = self.plot_item.addLegend()
            self.legend.addItem(curve, msg.name)
        elif isinstance(msg, MsgAddPoint):
            if msg.name not in self.curves:
                # Auto-create the series on left axis if unknown
                self._handle_msg(MsgAddSeries(msg.name, axis="left"))
            xb, yb = self.buffers[msg.name]
            xb.append(float(msg.t))
            yb.append(float(msg.y))
        elif isinstance(msg, MsgSetLabels):
            if msg.left is not None:
                self.plot_item.setLabel("left", msg.left)
            if msg.right is not None:
                self.plot_item.getAxis("right").setLabel(msg.right)
        elif isinstance(msg, MsgSetTitle):
            self.plot_widget.setWindowTitle(msg.title)
        elif isinstance(msg, MsgSetXRange):
            if msg.seconds is not None:
                # set a sliding window that moves with latest time
                # compute latest t across all buffers
                latest = None
                for xb, _ in self.buffers.values():
                    if xb:
                        latest = max(latest or -math.inf, xb[-1])
                if latest is not None and math.isfinite(latest):
                    self.plot_item.setXRange(latest - msg.seconds, latest)
            else:
                xmin = msg.xmin if msg.xmin is not None else self.plot_item.viewRange()[0][0]
                xmax = msg.xmax if msg.xmax is not None else self.plot_item.viewRange()[0][1]
                self.plot_item.setXRange(xmin, xmax)
        elif isinstance(msg, MsgClose):
            QtWidgets.QApplication.quit()
        else:
            pass


# --- Public façade used from non-Qt code ------------------------------------
class RealtimeGraph:
    def __init__(self, title: str = "Realtime Graph", tzname: str = "America/New_York", max_points: int = 100_000):
        self.q: Queue = mp.Queue()
        self.proc = _PlotProcess(self.q, title=title, tzname=tzname, max_points=max_points)
        self.proc.start()

    # Series & data
    def add_series(self, name: str, axis: str = "left", pen: Optional[Tuple[int, int, int]] = None):
        """Add a named series. axis in {"left","right"}. pen is optional RGB tuple."""
        self.q.put(MsgAddSeries(name=name, axis=axis, pen=pen))

    def add_point(self, name: str, t: float, y: float):
        """Push a single (t,y) point. t is Unix time in *seconds*."""
        self.q.put(MsgAddPoint(name=name, t=t, y=y))

    # Cosmetics / axes
    def set_labels(self, left: Optional[str] = None, right: Optional[str] = None):
        self.q.put(MsgSetLabels(left=left, right=right))

    def set_title(self, title: str):
        self.q.put(MsgSetTitle(title=title))

    def set_x_range(self, *, seconds: Optional[float] = None, xmin: Optional[float] = None, xmax: Optional[float] = None):
        """Either set a sliding window (seconds) or explicit [xmin,xmax] in Unix seconds."""
        self.q.put(MsgSetXRange(seconds=seconds, xmin=xmin, xmax=xmax))

    def close(self, timeout: float = 2.0):
        try:
            self.q.put(MsgClose())
            self.proc.join(timeout=timeout)
        finally:
            if self.proc.is_alive():
                self.proc.kill()


# --- Quick demo --------------------------------------------------------------
if __name__ == "__main__":
    g = RealtimeGraph(title="Realtime Demo", tzname="America/New_York", max_points=20_000)
    g.add_series("temp_degC", axis="left")
    g.add_series("pressure_kPa", axis="right")
    g.set_labels(left="°C", right="kPa")

    start = time.time()
    try:
        while True:
            t = time.time()
            # some fake signals
            y1 = 20 + 5 * math.sin(2 * math.pi * (t - start) / 10)
            y2 = 101 + 2 * math.cos(2 * math.pi * (t - start) / 7)
            g.add_point("temp_degC", t, y1)
            g.add_point("pressure_kPa", t, y2)
            g.set_x_range(seconds=60)  # rolling 60s window
            time.sleep(0.05)
    except KeyboardInterrupt:
        g.close()
