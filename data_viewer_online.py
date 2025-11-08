#!/usr/bin/env python3
"""
NPZ mid-price viewer with live zigzag label generation (pyqtgraph + Qt).

Features
- Loads NPZ with flexible 'features' / 'sampled' layouts.
- Plots mid-price on the left axis.
- Right axis shows labels that are recalculated on-the-fly from a modal dialog
  with sliders for zigzag params:
    * pct (fractional reversal threshold)
    * gain (fractional gain threshold from low to next high)
    * min_sep (minimum bar separation between extrema)
- Works with PyQt5 or PySide6.
- Qt5/Qt6 compatible .exec() and pen styles.

Usage examples:
  python viewer.py data.npz
  python viewer.py data.npz --bid-field bid --ask-field ask
  python viewer.py data.npz --bid-col 0 --ask-col 1
  python viewer.py data.npz --features-key X --labels-key y

Dependencies: pyqtgraph, PyQt5 (or PySide6), numpy
"""
from __future__ import annotations

import sys
import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Prefer PyQt5, fall back to PySide6
try:
    from PyQt5 import QtWidgets, QtCore
except Exception:  # pragma: no cover
    from PySide6 import QtWidgets, QtCore  # type: ignore

import pyqtgraph as pg
from label_zigzag import detect_missing, zigzag_extrema_sequence, compress_same_kind_extrema, pair_low_to_next_high_robust, label_bounces, make_events_array

# ----------------------------- Utilities -----------------------------

def _coerce_array(obj: Any):
    """Best-effort conversion of a stored object to a useful array/dict.
    Returns either a numpy array, a structured array, or a dict-like.
    """
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    return obj

def regenerate_labels(mid: np.ndarray, sampled: Optional[np.ndarray], pct: float, gain_thr: float, min_sep: int) -> np.ndarray:
    missing = detect_missing(sampled)

    # Build a working mid series for the zigzag by forward/back filling invalids
    mid_work = mid.copy()
    invalid = missing | np.isnan(mid_work) | ~(mid_work > 0)
    if invalid.any():
        # forward fill
        for i in range(1, len(mid_work)):
            if invalid[i] and not invalid[i - 1]:
                mid_work[i] = mid_work[i - 1]
        # back fill head if needed
        for i in range(len(mid_work) - 2, -1, -1):
            if invalid[i] and not invalid[i + 1]:
                mid_work[i] = mid_work[i + 1]

    # Detect extrema and pair LOW→next HIGH robustly
    seq = zigzag_extrema_sequence(mid_work, pct=pct, min_sep=min_sep)
    seq = compress_same_kind_extrema(seq, mid)
    pairs = pair_low_to_next_high_robust(seq, mid)

    # Point labels at LOWs
    labels, valid_idx, gains = label_bounces(mid, pairs, U=gain_thr, Type="Lo")

    # Remove any labels/pairs that land on missing rows (safety)
    if len(valid_idx):
        keep_mask = ~missing[valid_idx]
        valid_idx = valid_idx[keep_mask]
        gains = gains[keep_mask]
        # Keep only pairs whose LOW is kept
        low_set = set(valid_idx.tolist())
        pairs = [p for p in pairs if p[0] in low_set]
        # Clear labels at removed lows
        to_clear = [li for (li, _hi) in pairs if li not in low_set]
        if to_clear:
            labels[np.array(to_clear, dtype=np.int64)] = -1

    # Build events
    events = make_events_array(sampled, pairs, gains)


    # Brief console summary
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    inv = int((labels == -1).sum())
    print(f"[OK] | labels: +1={pos}, 0={neg}, -1={inv} | valid lows={len(valid_idx)} | pairs={len(pairs)}")

    return labels

# ------------------------------ Viewer -------------------------------

def build_app(
    npz_path: str,
    features_key: str,
    labels_key: str,
    pct,
    gain,
    sep,
    bid_field: str | None,
    ask_field: str | None,
    bid_col: int | None,
    ask_col: int | None,
    title: str | None,
    y2_min: float,
    y2_max: float,
    point_mode: bool,
):
    # High DPI polish (must be before QApplication construction)
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    # -------- Load --------
    data = np.load(npz_path, allow_pickle=True)
    available = set(data.files)

    features = _coerce_array(data[features_key]) if features_key in data else None
    labels_loaded = _coerce_array(data[labels_key]) if labels_key in data else None
    sampled = _coerce_array(data.get("sampled", None))

    # Prefer mid from 'sampled' if present, else compute from bid/ask
    mid_from_sampled: Optional[np.ndarray] = None
    if isinstance(sampled, np.ndarray) and getattr(sampled, "dtype", None) is not None and sampled.dtype.names:
        if "mid" in sampled.dtype.names:
            mid_from_sampled = sampled["mid"].astype(float)

    if mid_from_sampled is None:
        if features is None:
            available_s = ", ".join(sorted(available))
            raise KeyError(f"No '{features_key}' found and no 'sampled' present. Available keys: {available_s}")
        bid, ask = _extract_bid_ask(features, bid_field, ask_field, bid_col, ask_col)
        n = min(len(bid), len(ask))
        mid = (bid[:n] + ask[:n]) / 2.0
        sampled = None  # lacking spread/bid/ask validity checks
    else:
        mid = mid_from_sampled

    n = int(len(mid))
    x = np.arange(n)

    # -------- PyQt / pyqtgraph setup --------
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=title or f"NPZ Viewer: {npz_path}")
    win.resize(1200, 700)

    # Main plot (left y-axis) for mid-price
    p1 = win.addPlot(row=0, col=0)
    p1.setLabel('bottom', 'Index')
    p1.setLabel('left', 'Mid Price', units='')
    p1.showGrid(x=True, y=True, alpha=0.3)
    p1.addLegend()

    mid_curve = p1.plot(x, mid, pen=pg.mkPen(width=2), name='mid=(bid+ask)/2' if mid_from_sampled is None else 'mid')

    # Secondary axis and ViewBox for labels
    p2_axis = pg.AxisItem('right')
    p2_axis.setLabel('Labels')
    win.addItem(p2_axis, row=0, col=1)

    p2_vb = pg.ViewBox()
    p2_axis.linkToView(p2_vb)
    p2_vb.setXLink(p1)
    win.scene().addItem(p2_vb)

    # Plot labels placeholder (we set data after first generation)
    if point_mode:
        labels_item = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(200, 200, 255, 180))
    else:
        solid = getattr(QtCore.Qt, "SolidLine", getattr(QtCore.Qt.PenStyle, "SolidLine"))
        labels_item = pg.PlotDataItem(pen=pg.mkPen(style=solid, width=2))
    p2_vb.addItem(labels_item)

    # Keep the two views aligned on resize
    def update_views():
        p2_vb.setGeometry(p1.vb.sceneBoundingRect())
        p2_vb.linkedViewChanged(p1.vb, p2_vb.XAxis)

    p1.vb.sigResized.connect(update_views)
    update_views()

    # Set y-ranges
    p1.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
    p2_vb.setYRange(y2_min, y2_max, padding=0.05)

    # Nice integer ticks for the label axis
    ticks = [(i, str(i)) for i in range(int(np.floor(y2_min)), int(np.ceil(y2_max)) + 1)]
    p2_axis.setTicks([ticks])

    # Crosshair / mouse interaction
    vLine = pg.InfiniteLine(angle=90, movable=False)
    hLine = pg.InfiniteLine(angle=0, movable=False)
    p1.addItem(vLine, ignoreBounds=True)
    p1.addItem(hLine, ignoreBounds=True)
    label = pg.LabelItem(justify='right')
    win.addItem(label, row=1, col=0)

    # -------- Label generation state --------
    # Start with loaded labels if available; else compute default (pct=0.005, gain=0.003, min_sep=3)
    current_labels = np.asarray(labels_loaded).ravel() if labels_loaded is not None else regenerate_labels(
        mid, sampled, pct=pct, gain_thr=gain, min_sep=sep
    )

    if point_mode:
        labels_item.setData(x, current_labels)
    else:
        labels_item.setData(x=x, y=current_labels)

    # Throttle mouse move updates for heavy traces
    def mouseMoved(pos):
        if p1.sceneBoundingRect().contains(pos):
            mousePoint = p1.vb.mapSceneToView(pos)
            ix = int(round(mousePoint.x()))
            if 0 <= ix < n:
                curr_lab = current_labels[ix] if 0 <= ix < len(current_labels) else np.nan
                label.setText(f"x={ix}  mid={mid[ix]:.6g}  label={curr_lab:.6g}")
                vLine.setPos(ix)
                hLine.setPos(mid[ix])

    proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=lambda e: mouseMoved(e[0]))

    # -------- Modal with sliders --------
    class LabelParamsDialog(QtWidgets.QDialog):
        def __init__(self, parent=None, pct=0.005, gain=0.01, min_sep=1000):
            super().__init__(parent)
            self.setWindowTitle("Label Parameters")
            self.setModal(False)
            form = QtWidgets.QFormLayout(self)

            # pct slider+spin (0.0001 .. 0.0500)
            self.pct_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.pct_slider.setRange(1, 500)
            self.pct_spin = QtWidgets.QDoubleSpinBox()
            self.pct_spin.setDecimals(4)
            self.pct_spin.setRange(0.0001, 0.0500)
            self.pct_spin.setSingleStep(0.0001)

            # gain (0.0000 .. 0.0500)
            self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.gain_slider.setRange(0, 500)
            self.gain_spin = QtWidgets.QDoubleSpinBox()
            self.gain_spin.setDecimals(4)
            self.gain_spin.setRange(0.0, 0.0500)
            self.gain_spin.setSingleStep(0.0001)

            # min_sep (1 .. 200)
            self.sep_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.sep_slider.setRange(1, 5000)
            self.sep_spin = QtWidgets.QSpinBox()
            self.sep_spin.setRange(1, 5000)

            # wire up bi-directional sync
            def link(slider, spin, to_spin=lambda v: v / 10000.0, to_slider=lambda v: int(round(v * 10000))):
                slider.valueChanged.connect(lambda v: spin.setValue(to_spin(v)))
                spin.valueChanged.connect(lambda v: slider.setValue(to_slider(v)))

            link(self.pct_slider, self.pct_spin)
            link(self.gain_slider, self.gain_spin)
            self.sep_slider.valueChanged.connect(self.sep_spin.setValue)
            self.sep_spin.valueChanged.connect(self.sep_slider.setValue)

            # set initial values
            self.pct_spin.setValue(pct)
            self.gain_spin.setValue(gain)
            self.sep_spin.setValue(min_sep)

            # layout rows
            pct_row = QtWidgets.QHBoxLayout()
            pct_row.addWidget(self.pct_slider)
            pct_row.addWidget(self.pct_spin)
            gain_row = QtWidgets.QHBoxLayout()
            gain_row.addWidget(self.gain_slider)
            gain_row.addWidget(self.gain_spin)
            sep_row = QtWidgets.QHBoxLayout()
            sep_row.addWidget(self.sep_slider)
            sep_row.addWidget(self.sep_spin)
            form.addRow("Zigzag pct (fraction):", pct_row)
            form.addRow("Gain threshold (fraction):", gain_row)
            form.addRow("Min separation (bins):", sep_row)

            # buttons
            btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            btns.rejected.connect(self.reject)
            form.addRow(btns)

    params_action = QtWidgets.QAction("Labels…", win)
    params_action.setShortcut("Ctrl+L")
    win.addAction(params_action)

    dlg_holder: Dict[str, Optional[LabelParamsDialog]] = {"dlg": None}
    update_timer = QtCore.QTimer(win)
    update_timer.setSingleShot(True)
    update_timer.setInterval(100)  # debounce for smooth UI

    def schedule_update():
        update_timer.start()

    def do_update():
        nonlocal current_labels
        dlg = dlg_holder["dlg"]
        if dlg is None:
            return
        pct = float(dlg.pct_spin.value())
        gain_thr = float(dlg.gain_spin.value())
        min_sep = int(dlg.sep_spin.value())
        current_labels = regenerate_labels(mid, sampled, pct=pct, gain_thr=gain_thr, min_sep=min_sep)
        if point_mode:
            labels_item.setData(x, current_labels)
        else:
            labels_item.setData(x=x, y=current_labels)
        # keep right axis range stable as requested
        p2_vb.setYRange(y2_min, y2_max, padding=0.05)

    update_timer.timeout.connect(do_update)

    def open_params():
        if dlg_holder["dlg"] is None:
            dlg_holder["dlg"] = LabelParamsDialog(win)
            dlg_holder["dlg"].pct_spin.setValue(pct)
            dlg_holder["dlg"].gain_spin.setValue(gain)
            dlg_holder["dlg"].sep_spin.setValue(sep)
            # live update when any control changes
            for w in (
                dlg_holder["dlg"].pct_slider, dlg_holder["dlg"].pct_spin,
                dlg_holder["dlg"].gain_slider, dlg_holder["dlg"].gain_spin,
                dlg_holder["dlg"].sep_slider, dlg_holder["dlg"].sep_spin
            ):
                w.valueChanged.connect(schedule_update)
        dlg_holder["dlg"].show()
        dlg_holder["dlg"].raise_()
        dlg_holder["dlg"].activateWindow()

    params_action.triggered.connect(open_params)

    # Optional title
    if title:
        p1.setTitle(title)

    return app, win


def main(argv=None):
    parser = argparse.ArgumentParser(description="pyqtgraph NPZ dual-axis viewer with live zigzag labels")
    parser.add_argument('npz', help="Path to .npz file containing features/labels or sampled")
    parser.add_argument('--features-key', default='features', help="Key name for features inside the NPZ (default: features)")
    parser.add_argument('--labels-key', default='labels', help="Key name for labels inside the NPZ (default: labels)")
    parser.add_argument('--bid-field', help="Field/key name for bid inside features (when dict/structured)")
    parser.add_argument('--ask-field', help="Field/key name for ask inside features (when dict/structured)")
    parser.add_argument('--bid-col', type=int, help="Column index for bid if features is a plain 2D array")
    parser.add_argument('--ask-col', type=int, help="Column index for ask if features is a plain 2D array")
    parser.add_argument('--title', default=None, help="Optional window/plot title")
    parser.add_argument('--y2-min', type=float, default=-1.0, help="Right-axis (labels) min (default: -1)")
    parser.add_argument('--y2-max', type=float, default=5.0, help="Right-axis (labels) max (default: 5)")
    parser.add_argument('--points', action='store_true', help="Render labels as points instead of a line")
    parser.add_argument('--pct', type=float, help="pct")
    parser.add_argument('--gain', type=float, help="gain")
    parser.add_argument('--sep', type=int, help="pct")

    args = parser.parse_args(argv)

    try:
        app, win = build_app(
            npz_path=args.npz,
            features_key=args.features_key,
            labels_key=args.labels_key,
            bid_field=args.bid_field,
            ask_field=args.ask_field,
            bid_col=args.bid_col,
            ask_col=args.ask_col,
            title=args.title,
            y2_min=args.y2_min,
            y2_max=args.y2_max,
            point_mode=args.points,
            pct=args.pct,
            gain=args.gain,
            sep=args.sep,
        )
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(2)

    # Qt5: exec_(), Qt6: exec()
    runner = getattr(app, "exec", None) or getattr(app, "exec_", None)
    sys.exit(runner())


if __name__ == '__main__':
    main()
