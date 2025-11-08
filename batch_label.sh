#!/bin/bash
#find ./data -name *sampled_1s.npz -exec python label_zigzag.py --in {} --gain 0.01 --pct 0.005 --min_sep 1000 --Type Hi \;
#python split_labeled_datafiles.py --src ./data --out train-Hi val-Hi test-Hi --random --pct 70 15 15 --commit
#rm ./data/*labeled*
find ./data -name *sampled_1s.npz -exec python label_zigzag.py --in {} --gain 0.02 --pct 0.005 --min_sep 1000 --Type Lo \;
rm data/train-Lo/*
rm data/val-Lo/*
rm data/test-Lo/*
python split_labeled_datafiles.py --src ./data --out train-Lo val-Lo test-Lo --random --pct 70 15 15 --commit
rm ./data/*labeled*
