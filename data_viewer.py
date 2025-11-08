#!/usr/bin/env python3
"""
pyqtgraph NPZ viewer for mid-price and labels on dual y-axes.

Loads an .npz file that contains two entries:
  - features: a "sampled structure" that includes bid and ask
  - labels: numeric labels in the range [-1, 5]

It displays (bid + ask) / 2 as the mid-price on the left y-axis and labels
on a separate right y-axis with its own scale.

Flexible input handling:
- features may be a 2D numeric array, a structured/record array with fields,
  or a dict-like object stored in the npz. You can specify how to extract
  bid/ask via either field names or column indices.

Usage examples:
  python viewer.py data.npz
  python viewer.py data.npz --bid-field bid --ask-field ask
  python viewer.py data.npz --bid-col 0 --ask-col 1
  python viewer.py data.npz --features-key X --labels-key y --bid-field best_bid --ask-field best_ask

Dependencies: pyqtgraph, PyQt5 (or PySide6), numpy
"""
from __future__ import annotations
import sys
import argparse
import numpy as np

# Prefer PyQt5, fall back to PySide6
try:
    from PyQt5 import QtWidgets, QtCore
except Exception:  # pragma: no cover
    from PySide6 import QtWidgets, QtCore # type: ignore
import pyqtgraph as pg


def _coerce_array(obj):
    """Best-effort conversion of a stored object to a useful array/dict.
    Returns either a numpy array, a structured array, or a dict-like.
    """
    # If it's a 0-d object array holding a dict or ndarray, unwrap it
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    return obj


def _extract_bid_ask(features, bid_field: str | None, ask_field: str | None,
                     bid_col: int | None, ask_col: int | None):
    """Extract bid and ask arrays from a flexible "features" container.

    - If `features` is a dict-like, use dict keys first (bid_field/ask_field if given, else best-guess).
    - If it's a structured/record array (dtype.names), use field names.
    - Else assume 2D numeric and use column indices.
    """
    # If dict-like
    if isinstance(features, dict):
        keys = list(features.keys())
        def pick_key(name_hint: str | None):
            if name_hint and name_hint in features:
                return name_hint
            # fuzzy find
            lowered = {k.lower(): k for k in keys}
            for needle in ("bid", "best_bid", "b"):
                if needle in lowered:
                    return lowered[needle]
            return None
        bid_k = pick_key(bid_field)
        ask_k = pick_key(ask_field)
        if bid_k is None or ask_k is None:
            raise KeyError("Could not locate 'bid'/'ask' keys in features dict. Use --bid-field and --ask-field.")
        bid = np.asarray(features[bid_k], dtype=float).ravel()
        ask = np.asarray(features[ask_k], dtype=float).ravel()
        return bid, ask

    # Structured / record array
    if isinstance(features, np.ndarray) and features.dtype.names:
        names_lower = {n.lower(): n for n in features.dtype.names}
        def pick_field(name_hint: str | None, candidates):
            if name_hint:
                # allow exact or case-insensitive match
                if name_hint in features.dtype.names:
                    return name_hint
                if name_hint.lower() in names_lower:
                    return names_lower[name_hint.lower()]
            for c in candidates:
                if c in names_lower:
                    return names_lower[c]
            return None
        bid_name = pick_field(bid_field, ("bid", "best_bid", "b"))
        ask_name = pick_field(ask_field, ("ask", "best_ask", "a"))
        if bid_name is None or ask_name is None:
            raise KeyError("Could not locate 'bid'/'ask' fields in structured features. Use --bid-field/--ask-field.")
        bid = np.asarray(features[bid_name], dtype=float).ravel()
        ask = np.asarray(features[ask_name], dtype=float).ravel()
        return bid, ask

    # 2D numeric array; use column indices
    arr = np.asarray(features)
    if arr.ndim < 2 or arr.shape[1] < 2:
        raise ValueError("Features must be dict/structured or a 2D array with >=2 columns for bid/ask.")
    if bid_col is None or ask_col is None:
        raise ValueError("For plain 2D arrays, specify --bid-col and --ask-col.")
    bid = arr[:, int(bid_col)].astype(float)
    ask = arr[:, int(ask_col)].astype(float)
    return bid, ask


def build_app(npz_path: str,
              features_key: str,
              labels_key: str,
              bid_field: str | None,
              ask_field: str | None,
              bid_col: int | None,
              ask_col: int | None,
              title: str | None,
              y2_min: float,
              y2_max: float,
              point_mode: bool,
              ):
    # Load
    data = np.load(npz_path, allow_pickle=True)
    if features_key not in data or labels_key not in data:
        available = ", ".join(sorted(data.files))
        raise KeyError(f"NPZ must contain '{features_key}' and '{labels_key}'. Found: {available}")

    features = _coerce_array(data[features_key])
    labels = _coerce_array(data[labels_key])

    # Extract bid/ask
    bid, ask = _extract_bid_ask(features, bid_field, ask_field, bid_col, ask_col)
    n = min(len(bid), len(ask))
    mid = (bid[:n] + ask[:n]) / 2.0

    labels = np.asarray(labels).ravel()
    if len(labels) != n:
        m = min(n, len(labels))
        mid = mid[:m]
        labels = labels[:m]
        n = m

    x = np.arange(n)

    # PyQt / pyqtgraph setup
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title=title or f"NPZ Viewer: {npz_path}")
    win.resize(1200, 700)

    # Main plot (left y-axis) for mid-price
    p1 = win.addPlot(row=0, col=0)
    p1.setLabel('bottom', 'Index')
    p1.setLabel('left', 'Mid Price', units='')
    p1.showGrid(x=True, y=True, alpha=0.3)
    p1.addLegend()

    mid_curve = p1.plot(x, mid, pen=pg.mkPen(width=2), name='mid=(bid+ask)/2')

    # Secondary axis and ViewBox for labels
    p2_axis = pg.AxisItem('right')
    p2_axis.setLabel('Labels [-1..5]')
    win.addItem(p2_axis, row=0, col=1)

    p2_vb = pg.ViewBox()
    p2_axis.linkToView(p2_vb)
    p2_vb.setXLink(p1)
    win.scene().addItem(p2_vb)

    # Plot labels: allow line or scatter
    if point_mode:
        labels_item = pg.ScatterPlotItem(x, labels, size=6, pen=None, brush=pg.mkBrush(200, 200, 255, 180))
    else:
        labels_item = pg.PlotDataItem(x, labels, pen=pg.mkPen(style=QtCore.Qt.SolidLine, width=2))

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

    def mouseMoved(evt):
        pos = evt
        if p1.sceneBoundingRect().contains(pos):
            mousePoint = p1.vb.mapSceneToView(pos)
            ix = int(round(mousePoint.x()))
            if 0 <= ix < n:
                label.setText(f"x={ix}  mid={mid[ix]:.6g}  label={labels[ix]:.6g}")
                vLine.setPos(ix)
                hLine.setPos(mid[ix])
    p1.scene().sigMouseMoved.connect(mouseMoved)

    # Final touches
    if title:
        p1.setTitle(title)

    return app, win


def main(argv=None):
    parser = argparse.ArgumentParser(description="pyqtgraph NPZ dual-axis viewer for mid-price and labels")
    parser.add_argument('npz', help="Path to .npz file containing features and labels")
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
        )
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(2)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
