#!/usr/bin/env bash
# v5 live additions (e7v5_liveblock.py), run as root in WSL Ubuntu:
#  B1 benign-only scenario: 5 configurations x 5 repeats (configurations rotated between repeats)
#  B3 storm-intensity sweep: rate rule always on, 5 intensities x 5 repeats
cd "$(dirname "$0")"
CONFIGS=(lab commissioned rate_always ks_rate full)
for rep in 1 2 3 4 5; do
  for i in 0 1 2 3 4; do
    cfg=${CONFIGS[$(( (i + rep) % 5 ))]}
    [ -f "/mnt/d/shield_run/outputs_v5/liveblock/benign/${cfg}_r${rep}.json" ] && continue
    python3 e7v5_liveblock.py benign "$cfg" "$rep" 2>/dev/null | tail -1
  done
done
for rep in 1 2 3 4 5; do
  for s in storm_g050 storm_g033 storm_g025 storm_g018 storm_g012; do
    [ -f "/mnt/d/shield_run/outputs_v5/liveblock/$s/rate_always_r${rep}.json" ] && continue
    python3 e7v5_liveblock.py "$s" rate_always "$rep" 2>/dev/null | tail -1
  done
done
echo ALL_DONE
