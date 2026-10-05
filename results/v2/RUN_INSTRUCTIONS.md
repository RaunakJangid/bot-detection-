# SHIELD-IoT v2 run: instructions

The rules from the v1 run apply unchanged (hard rules, the interpreter rule, the disk rule, the topology rule). They are copied below the v2 prompt.

## v2 prompt (user, 2026-10-04)
New pipeline version v2 (commit e1f85e4). Run it unattended with the same rules as last time
(RUN_INSTRUCTIONS.md hard rules, the interpreter rule, the disk rule, the topology rule).
1. git pull, then .venv\Scripts\python.exe -m pytest -q (expect 57 passed). Never use uv run / uv sync.
2. In configs/paths.yaml keep data_dir D:/shield_run/data and set outputs_dir: "D:/shield_run/outputs_v2"
   (a fresh results folder; never touch D:/shield_run/outputs, which holds v1). No byte-order mark.
   Copy D:\shield_run\outputs\RUN_STATE.md's location notes into a new D:\shield_run\outputs_v2\RUN_STATE.md.
3. Smoke test: scripts\run_all.py --smoke must end with "Smoke check passed". GPU check: at least 8 GB free.
4. No quick pass this time (tuning is built in, on validation only). Launch scripts\run_all.py detached,
   no --force, log to outputs_v2\logs\. Do not edit any experiment config during the run.
   Allowed engineering fixes only: CUDA OOM or heavy RAM paging -> lower kd.yaml jobs 3 -> 2 and relaunch;
   pre-read large .npy files to warm the RAM cache when streaming from D: is slow.
   06_latency must run with the CPU otherwise idle (it does in run_all).
5. When it finishes, write D:\shield_run\outputs_v2\FINAL_REPORT_v2.md for a beginner. Mean +- std over seeds,
   and for each item compare v2 with v1 (D:\shield_run\outputs\FINAL_REPORT.md):
   A teacher scores; B the tuning choice (objective, teacher/assistant), and whether KD now beats scratch on
   ciciot family/class34 and botiot category; C ranking and per-task k/size, and the student-teacher gap;
   D low-data Wilcoxon (shield vs kd_shap at 0.1/1/10%) and agreement; E CORAL/DANN vs the earlier
   normalisations; F placement ablation, tuned settings, gap and ranks (hybrid vs SA); G coupling on all
   3 topologies with their SLA; H int8. Plus the CD diagrams and honest limitations.
6. Copy FINAL_REPORT_v2.md and every CSV in outputs_v2\paper\tables\ into the repo folder results\v2\,
   commit and push them (reports and small tables only, never data/ or model files).
Log everything in RUN_STATE.md, then stop the loop.

---------------------------------------------------------------------------------------------------
## v1 instructions and amendments (still in force; "outputs/" now means D:\shield_run\outputs_v2)
You are running the SHIELD-IoT experiment pipeline UNATTENDED for up to 3 days. The user is away and
will not answer questions: never ask, decide conservatively using the rules below, and log every
decision. This prompt is re-sent on every /loop wake-up, so always work from the state file.

## State file
outputs/RUN_STATE.md: create it if missing. It records the current PHASE, process IDs, start times,
log paths, decisions made, and problems found. Read it FIRST on every wake-up and continue from the
recorded phase. Append; never erase history.

## How to run long jobs
- Never run multi-hour jobs as foreground tool calls. Launch them detached, e.g.
  Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList "scripts\run_all.py" `
    -RedirectStandardOutput outputs\logs\<name>.log -RedirectStandardError outputs\logs\<name>.err `
    -NoNewWindow -PassThru
  and record the PID in RUN_STATE.md.
- On each wake-up: check the PID is alive (Get-Process -Id), tail the logs, look for
  Traceback / Error / FAILED / CUDA out of memory.
- If a process died: diagnose from the log, fix the cause, relaunch the same command. All scripts are
  resumable and skip finished runs.
- Schedule the next wake-up 30-60 min ahead while jobs run (shorter only right after a launch, to
  catch early crashes).

## Hard rules (never break these)
- Never change the scientific protocol: dataset definitions, the strict feature removal (IAT, seq),
  the splits, the de-duplication, or the test data. Never look at test-set results to choose any
  setting. Tuning is allowed only where written below, and only with VALIDATION metrics.
- Never delete data/, the dataset zips, or outputs/ wholesale. Never run run_all.py with --force.
  Delete a specific run directory only when a rule below says so.
- Allowed code changes: bug and crash fixes. After any code change, `uv run pytest` must pass. Commit
  with a clear message and push code/config only (never data/ or outputs/).
- Engineering fixes allowed without asking: CUDA OOM -> lower kd.yaml `jobs` or a batch size; CPU
  contention -> fewer placement workers; disk-space problems -> report and stop.

## PHASE 1: Preflight
1. Check configs/paths.yaml points at three existing zips (CICIoT2023, Bot-IoT, CICIoMT2024).
   If any is missing, write that to RUN_STATE.md and stop.
2. Run `uv run pytest`. Check the GPU:
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
3. Run the smoke test detached: scripts\run_all.py --smoke. It must end with "Smoke check passed".
4. (User amendment, 2026-10-02 ~02:24) Check GPU memory: nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   If less than 8 GB is free, do NOT kill other users' processes. Log a WARN in RUN_STATE.md with
   the process list and continue (data falls back to host-memory streaming: slower but correct).
   Re-check at the start of PHASE 3 and log the value.
(User amendment) Before launching: make sure no other pipeline run (e.g. another Claude session) is writing to
the same outputs; let it finish or stop it first.

## PHASE 2: Quick pass (seed 0, about 6-8 h)
1. Set the quick-pass configs (record the originals in RUN_STATE.md):
   - configs/teacher.yaml   seeds: [0]
   - configs/kd.yaml        seeds: [0]
   - configs/datasets.yaml  seeds: [0] for xciciot and xciciomt
   - configs/placement.yaml runs: 5
2. Launch scripts\run_all.py detached (no --smoke). The CPU must be otherwise idle while
   06_latency.py runs, so its timings are valid.
3. When it finishes, write outputs/QUICK_PASS_REPORT.md with a PASS / WARN / FAIL line for each
   check below, plus the key numbers. Sources: outputs/data_report_*.json, outputs/leakage/,
   outputs/paper/tables/, outputs/placement/, outputs/coupled/, outputs/cross/, and the per-run
   metrics.json files.
   a. Data: row counts per split for every dataset; no main dataset is missing a class in train;
      de-dup statistics (removed duplicates and conflicting rows) are reported.
   b. Leakage audit: IAT (CIC) and seq (Bot-IoT) rank as suspicious single features. Name any OTHER
      feature whose single-feature macro-F1 is at least 80% of the all-features score as a suspect.
      Report it only; do not remove it.
   c. Inflation: macro-F1 under the standard protocol vs the strict protocol, per task.
   d. Teachers: test macro-F1 per task vs XGBoost. WARN if a teacher is more than 5 points below
      XGBoost, or early-stopped at epoch <= 3. Check that IAT/seq are not among the strict SHAP features.
   e. KD ablation: shield vs kd_shap vs kd_all vs scratch_all vs random/mi/variance, int8 drop,
      SHAP fidelity, parameter count, and the dtree/logreg baselines. WARN if dtree or logreg is
      within 1 point of shield on a task.
   f. Latency: student vs teacher single-core throughput; int8 vs fp32.
   g. Placement: optimality gap vs brute force on small instances (the hybrid should be at or near 0%);
      average ranks; runtime. WARN if the hybrid is outside the top 3 on the large topologies
      (Cogentco, Kdl, syn500).
   h. Coupled: the teacher needs more controllers than shield at every load level.
   i. Cross-dataset: in-domain vs zero-shot drop, target-normalisation vs source-normalisation,
      few-shot fine-tuned vs scratch at each fraction, never-seen attack detection rates, SHAP
      agreement. WARN where fine-tuned is not better than scratch.
   j. Every figure and table exists in outputs/paper/.
4. The only allowed tuning (validation only, decide automatically):
   If shield's best VALIDATION macro-F1 (max of history[].val_macro_f1 in metrics.json) is below
   kd_shap's on at least 5 of the 8 main tasks, sweep the attribution weight gamma over {0.03, 0.3}
   (current default 0.1):
   - Add temporary variants to configs/kd.yaml copying `shield` with gamma 0.03 / 0.3, named
     shield_g003 / shield_g03.
   - Run them on the main datasets with seed 0: 05_distill.py --dataset main --variants shield_g003 shield_g03 --seeds 0
   - Choose the gamma with the best MEAN validation macro-F1 across the 8 main tasks (including 0.1).
   - If it isn't 0.1: set kd.yaml loss.gamma to it, delete only outputs/kd/*/shield_s0 and
     outputs/kd/*/shield_k*_s0 directories, re-run 05_distill.py --seeds 0 and the --k-sweep, then
     re-run 06_latency.py with the CPU idle. Remove the temporary variants from kd.yaml either way.
     Log the decision.
   Change nothing else, whatever the quick pass shows. Record WARNs for the user to decide on later.

## PHASE 3: Full run (about 2-2.5 days)
1. Restore the full configs: teacher seeds [0,1,2,3,4], kd seeds [0,1,2,3,4], cross datasets seeds
   [0,1,2], placement runs 30. Seed-0 results are reused automatically.
2. Run two detached processes IN PARALLEL to save a day:
   A) GPU chain, run sequentially (each step only adds missing runs):
      03_train_teacher.py, 03b_xgboost.py, 03c_baselines.py, 04_shap.py, 05_cache_teacher.py,
      05_distill.py, 05_distill.py --k-sweep, 09_cross_dataset.py
      Write a small .ps1 that runs them in order and stops on the first failure, and launch it.
   B) CPU: 07_placement.py --workers 12   (uses the quick pass's outputs/latency/latency.json)
   Do NOT run 06_latency.py while B is running: the timings would be wrong. The quick-pass
   latency.json stays the source of truth.
3. When BOTH are finished: run 07b_coupled.py, then 08_make_figures.py.

## PHASE 4: Final report
Write outputs/FINAL_REPORT.md for the user (a beginner to ML and networking). Use plain language and
mean ± std over seeds. Cover:
- the headline results for each of the three contributions (SHAP-guided KD, cross-dataset transfer
  and recovery, hybrid EHO-ACO placement plus coupling);
- the leakage findings and the inflation numbers;
- every WARN from the quick pass and whether it persisted in the full run;
- the critical-difference diagram results (Friedman p-value, which methods are statistically tied);
- total runtime and any decisions or fixes you made;
- honest limitations;
- which paper tables/figures to use, with their paths.
Then set PHASE: DONE in RUN_STATE.md, push any code/config fixes, and stop scheduling wake-ups.

## Locations (user amendment 2026-10-02 02:27, URGENT disk-space fix)
data_dir = D:/shield_run/data and outputs_dir = D:/shield_run/outputs (set in configs/paths.yaml).
Every "outputs/..." or "data/..." path above means D:\shield_run\outputs\... / D:\shield_run\data\...
The state file is D:\shield_run\outputs\RUN_STATE.md. Run scripts from the repo C:\Users\Prakash\shield.
Disk rule: check free space on D: (and C:) on every wake-up; below 8 GB, stop the run cleanly and record it.

## Interpreter rule (user amendment 2026-10-02 ~02:35)
For the rest of this run, never use `uv run` or `uv sync`: they can swap in the CPU-only PyTorch build.
Always call the venv interpreter directly: .venv\Scripts\python.exe (scripts and pytest), e.g.
`.venv\Scripts\python.exe -m pytest -q`. If torch ever reports cuda.is_available() == False, reinstall with
`python -m uv sync --extra cu128`, log it in RUN_STATE.md, then continue.

## Placement download rule (user amendment 2026-10-02 ~02:40)
If 07_placement.py fails because a Topology Zoo download fails, don't change the topology list. Run the
remaining steps (07b_coupled.py, 09_cross_dataset.py, 08_make_figures.py) manually so they aren't
blocked, retry 07_placement.py on later wake-ups, and log it in RUN_STATE.md.
