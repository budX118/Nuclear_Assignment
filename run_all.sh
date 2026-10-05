#!/usr/bin/env bash
# Reproduces every result of the assignment.  Total time on a 4-core laptop: ~1 min (10^6) + ~1-2 h (10^9).
set -e
pip install -r requirements.txt
python mc_reactor.py --n 1e6 --out results_1e6                 # 1 million neutrons per case   (~15 s)
python mc_reactor.py --scan --scan-n 1e6 --out results_scan    # k_inf versus moderator amount (~2 min)
python mc_reactor.py --n 1e6 --coarse --out results_1e6_coarse # only the 50-point table        (~15 s)
python mc_reactor.py --n 1e9 --out results_1e9 --resume        # 1 billion neutrons per case    (checkpointed)
python analyze.py --main results_1e9 --small results_1e6 --scan results_scan/scan.json --coarse results_1e6_coarse --out analysis
python crosscheck_deterministic.py --mc results_1e9 --out analysis
