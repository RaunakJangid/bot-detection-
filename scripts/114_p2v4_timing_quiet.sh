#!/bin/bash
# P2 v4: quiet-machine timing reruns (T1b IoT-23 = scripts/102 fixed, T1 CIC = scripts/98), 3 independent runs each.
# Waits for: both EHO-ACO ablations analysed, the label audits finished (flag file), and CPU idle (< 10% over 5 s, checked
# 3 times in a row). Before every run a process-level snapshot (scripts/115) is saved next to its result. Each run's JSON/log is kept under its own name; superseded runs are never overwritten:
#   capacity_curve_iot23_INVALID_run1.*  (NaN threshold + 84-88% load; superseded)
#   capacity_curve_run2_load35pct.json   (CIC, 30-35% load; superseded as baseline)
P=/d/shield_run/outputs_v5/p2; S=/c/Users/Prakash/shield
until [ -f $P/ablation_shift_stats.json ] && [ -f /d/shield_run/outputs_v5/iot23_audit/audits_done.flag ]; do sleep 60; done
idle() { $S/.venv/Scripts/python.exe -c "import psutil,sys; sys.exit(0 if psutil.cpu_percent(5) < 10 else 1)"; }
ok=0; until [ $ok -ge 3 ]; do if idle; then ok=$((ok+1)); else ok=0; sleep 30; fi; done
cd $S
for r in 1 2 3; do
  date "+%H:%M T1b quiet run $r start"
  .venv/Scripts/python.exe scripts/115_proc_snapshot.py $P/capacity_curve_iot23_quiet_r${r}_procs.json
  .venv/Scripts/python.exe -W ignore scripts/102_p2v4_capacity_curve_iot23.py > $P/capacity_curve_iot23_quiet_r$r.log 2>&1
  mv $P/capacity_curve_iot23.json $P/capacity_curve_iot23_quiet_r$r.json; tail -7 $P/capacity_curve_iot23_quiet_r$r.log
done
for r in 1 2 3; do
  date "+%H:%M T1 CIC quiet run $r start"
  .venv/Scripts/python.exe scripts/115_proc_snapshot.py $P/capacity_curve_quiet_r${r}_procs.json
  .venv/Scripts/python.exe -W ignore scripts/98_p2v4_capacity_curve.py > $P/capacity_curve_quiet_r$r.log 2>&1
  cp $P/capacity_curve.json $P/capacity_curve_quiet_r$r.json; tail -14 $P/capacity_curve_quiet_r$r.log
done
date "+%H:%M timing done"
