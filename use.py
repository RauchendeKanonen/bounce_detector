from rt_plot import RealtimeGraph
import time, random

g = RealtimeGraph(title="Sensors", tzname="America/New_York", max_points=100_000)
g.add_series("temp_degC", axis="left")
g.add_series("pressure_kPa", axis="right")
g.set_labels(left="°C", right="kPa")
g.set_x_range(seconds=300)  # rolling 5-minute window

while True:
    t = time.time()  # Unix seconds
    g.add_point("temp_degC", t, 20 + random.random()*5)
    g.add_point("pressure_kPa", t, 101 + random.random()*2)
    time.sleep(0.05)  # 20 Hz
