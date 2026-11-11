#!/usr/bin/env bash
# Cycle 1 pilot: 12 pre-registered runs (R0, S1, S2, T1 x 3 repeats), sequential, conditions interleaved per repeat.
cd /mnt/c/Users/Prakash/shield/scripts
for rep in 1 2 3; do
  for c in R0 S1 S2 T1; do
    [ -f "/mnt/d/shield_run/papers/novelty/runs/cycle_1/${c}_r${rep}.json" ] && continue
    python3 x8_sensor_pilot.py "$c" "$rep" 2>&1 | grep -v sch_htb | tail -1
  done
done
echo PILOT_DONE
