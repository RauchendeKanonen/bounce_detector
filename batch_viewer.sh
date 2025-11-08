#!/bin/bash
find ./data -name "*events.npz" -exec python data_viewer_online.py --features-key sampled {} \;
