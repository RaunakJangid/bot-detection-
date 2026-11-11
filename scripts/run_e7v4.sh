#!/usr/bin/env bash
# Matched live ablation (e7v4_liveblock.py): 5 repeats x 3 scenarios x 5 configurations, configurations interleaved
# within each repeat and rotated between repeats so no configuration always runs first. Run as root in WSL Ubuntu.
cd "$(dirname "$0")"
CONFIGS=(lab commissioned rate_always ks_rate full)
for rep in 1 2 3 4 5; do
  for scen in attacks bursts distributed; do
    for i in 0 1 2 3 4; do
      cfg=${CONFIGS[$(( (i + rep) % 5 ))]}
      [ -f "/mnt/d/shield_run/outputs_v4/liveblock/$scen/${cfg}_r${rep}.json" ] && continue
      python3 e7v4_liveblock.py "$scen" "$cfg" "$rep" 2>/dev/null | tail -1
    done
  done
done
echo ALL_DONE
