# SHIELD-IoT unattended run: state log

Instructions: outputs/RUN_INSTRUCTIONS.md (verbatim copy of the user's run prompt). Re-read it on every wake-up.
Git: C:\Users\Prakash\AppData\Local\Programs\Git\cmd\git.exe (not on PATH). uv: `python -m uv` (system python).
Repo commit at start: e62c5b0 (strict leakage protocol, CICIoMT2024, cross-dataset).

## Current
PHASE: DONE (2026-10-02 19:50). Final report: D:\shield_run\outputs\FINAL_REPORT.md. Quick-pass report: D:\shield_run\outputs\QUICK_PASS_REPORT.md.
No jobs running. No code/config changes to push (only the local configs/paths.yaml differs from git). Wake-ups stopped.

## Log (append only)

### 2026-10-02 02:20: Phase 1 started
- Zips (configs/paths.yaml), all exist:
  - ciciot_zip  = D:/bot-datasets/archive (1).zip   (0.47 GB, 2.32 GB uncompressed, 3 CSVs; looks like a CICIoT2023 subset)
  - botiot_zip  = D:/bot-datasets/archive.zip       (1.17 GB, 15.0 GB uncompressed, 75 CSVs)
  - ciciomt_zip = C:/Users/Prakash/Downloads/ciciomt2024.zip (0.26 GB, 2.28 GB uncompressed, 72 CSVs)
- pytest: 41 passed (run 2026-10-02 ~02:10 on e62c5b0, before this log existed).
- GPU: True, NVIDIA GeForce RTX 3060. VRAM free only 2.62 / 12.0 GB.
- Disk: C: (SSD, holds data/ and outputs/) 37.9 GB free; D: (HDD) 643 GB free.
- Smoke test: launched 02:18 as a tool-background job (not detached), log
  %TEMP%\claude\...\scratchpad\smoke3.log. NOTE: it was started with `run_all.py --smoke --force`
  just BEFORE these instructions arrived; --force only affects data_smoke/ and outputs_smoke/.
  All later launches follow the rules (no --force).

### Known risks (recorded, not acted on)
- R1 VRAM: about 32 stale python.exe processes (started 2026-09-16..25, another account or elevated, command line
  hidden) hold about 9-10 GB of VRAM. Cannot inspect or kill them (permissions). Effect: large datasets will
  not fit on the GPU (Batcher puts data on GPU only if < 60% of free VRAM) and fall back to memmap
  reads from the SSD, so they run slower but correctly. CUDA OOM risk with kd.yaml jobs: 3. If an OOM happens,
  lower jobs (allowed fix).
- R2 Disk: full run estimated about 25-35 GB on C: (rough). Rule: disk-space problems mean report and stop.
  Check free space on every wake-up. If C: drops below 8 GB, record it and stop launching new work.

### 2026-10-02 02:24: Phase 1 complete
- Smoke test (strict-protocol code e62c5b0): PASSED. All 15 steps, 5.0 min, "Smoke check passed: every expected output exists."
  Only noise: sklearn logreg "failed to converge after 50 iterations" (smoke caps logreg_max_iter at 50).
- User amendment received (GPU check threshold 8 GB free, warn-and-continue; no concurrent pipeline runs).
  Copied into RUN_INSTRUCTIONS.md, Phase 1 step 4.
- GPU check (step 4): nvidia-smi says 1799 / 12288 MiB used, so about 10.5 GB free (>= 8 GB): OK, no WARN.
  The stale python.exe processes behind risk R1 are gone (Get-Process/Win32_Process find no python.exe).
  R1 resolved for now; re-check at Phase 3 start.
- Concurrency check: no python/uv processes and no PowerShell pipeline scripts running, so nothing else writes outputs/.
  The earlier smoke run (tool-background) had already finished.
- CPU at idle: 3% load, 20 logical cores.

### 2026-10-02 02:24: Phase 2 started (quick pass)
- Original configs (restore these in Phase 3):
  - configs/teacher.yaml   seeds: [0, 1, 2, 3, 4]
  - configs/kd.yaml        seeds: [0, 1, 2, 3, 4]
  - configs/datasets.yaml  xciciot.seeds: [0, 1, 2], xciciomt.seeds: [0, 1, 2]
  - configs/placement.yaml runs: 30
- Quick-pass values set: teacher [0], kd [0], xciciot/xciciomt [0], placement runs 5. Verified with load_config/yaml.
  (PowerShell 5.1 Set-Content had added a UTF-8 BOM; rewrote the files BOM-free and re-verified.)
  These config edits are temporary and NOT committed.
- configs/paths.yaml is local-only (machine paths), never committed.
- Launched: Start-Process .venv\Scripts\python.exe scripts\run_all.py (no --smoke, no --force), PID 141176.
  First 45 s fine: CICIoT2023 ingest train 5,491,971 / validation 1,176,851 / test 1,176,851 rows.
- Disk at launch: C: 37.9 GB free.

### 2026-10-02 02:27-02:30: Storage move to D: (user instruction "URGENT disk-space fix")
- Reason (user's estimate): the run needs about 45-50 GB (interim ~5, processed ~25 incl. botiot_std ~12, teacher caches ~17);
  C: had 34.8 GB free. D: had 643.4 GB free (>= 150 GB), so the move path applies.
- 02:28:02 stopped the quick pass: PIDs 141176 (run_all), 141748, 160800, 132904 (02_preprocess, which was partway through
  Bot-IoT; CICIoT2023 and CICIoMT2024 were already processed). Confirmed no shield python.exe left.
  CICIoMT de-dup stats seen before the stop: rows_in 8,775,013, duplicates dropped 3,391,891, conflicting rows dropped 859,567,
  rows_out 5,380,080 (train 3,846,494 / val 646,417 / test 887,169).
- Moved with robocopy /MOVE (PS 5.1 Move-Item cannot move folders between drives): data\ -> D:\shield_run\data (105 files, 3.09 GB),
  outputs\ -> D:\shield_run\outputs (6 files). 0 failures. The C:\...\shield\data and \outputs folders are now gone.
- configs/paths.yaml: data_dir "D:/shield_run/data", outputs_dir "D:/shield_run/outputs". No BOM; Paths() confirmed to load them.
- NOTE: Paths.topologies is hardcoded to <repo>/data/topologies (C:), not data_dir. Tiny files, left as is; the code will
  recreate C:\Users\Prakash\shield\data\topologies when placement runs. Not a problem.
- git pull: 6628d75 "Read YAML configs as utf-8-sig" (src/shield/utils/config.py only). `python -m uv run pytest`: 41 passed.
  Checked afterwards that `uv run` had not swapped torch: 2.11.0+cu128, cuda True.
- Partial Bot-IoT preprocessing had no meta.json, so it is redone; ciciot/ciciomt have meta.json and are skipped.
- 02:29:14 relaunched: Start-Process .venv\Scripts\python.exe scripts\run_all.py (no --force), PID 88768,
  log D:\shield_run\outputs\logs\quick_pass_2.log. After 60 s: ingest skipped, botiot preprocess gathering 16/74. OK.
  (First launch's logs: D:\shield_run\outputs\logs\quick_pass.log/.err.)
- Disk after move: see next wake-up. Now watching D: (HDD) and C:; stop rule (< 8 GB free) applies to the drive holding data/outputs (D:).
- Performance note: D: is an HDD. Data that fits in GPU memory (about 10 GB free now) is loaded once, so this costs little;
  data that does not fit streams from the HDD, which is slower but correct.

### 2026-10-02 02:33: User amendment: never use uv run / uv sync; use .venv\Scripts\python.exe for scripts and pytest (copied into RUN_INSTRUCTIONS.md). Loop restarted as '/loop continue as the plan ...'.

### 2026-10-02 02:40: Topology Zoo download problem, resolved ahead of time
- User amendment (placement download rule) copied into RUN_INSTRUCTIONS.md.
- Pre-checked the downloads: www.topology-zoo.org (http and https) times out from this machine (DNS resolves to 129.127.10.249, TCP:80 fails).
  raw.githubusercontent.com also times out. github.com, pypi and google work. So 07_placement.py would have failed at download_zoo.
- First try: jsDelivr mirror of github sk2/topologyzoo /sources/*.graphml. REJECTED: those are pre-geocoding source drawings
  (x/y only, no Latitude/Longitude), so load_zoo produced an empty graph. Deleted them immediately so the code cannot pick them up.
- Fix: official long-term Zoo store github.com/mroughan/InternetTopologyZoo (M. Roughan, Zoo co-author; figshare DOI 10.25909/30153949),
  commit 51118849b995 (2026-05-24), files graphml/<name>.graphml, fetched via a git sparse clone and placed in the cache path the code
  reads: C:\Users\Prakash\shield\data\topologies\. download_zoo() sees the cached file and skips the network. Topology list UNCHANGED;
  no code change.
- Validated with load_zoo: Abilene 11 nodes / 14 edges, Geant2012 37/58 (raw 40), Cogentco 180/210 (raw 197), Kdl 709/815 (raw 754).
  Hashes (compare with the laptop's copies if it downloaded from topology-zoo.org):
  Abilene.graphml  sha256 8CD694280D98B336
  Cogentco.graphml  sha256 FCA525C7484C1044
  Geant2012.graphml  sha256 8487B2FB091D53F1
  Kdl.graphml  sha256 F0EF1032BB89695C

### 2026-10-02 02:54: Check: healthy. PID 88768 alive, no errors. Preprocess 392 s and leakage audit 30 s done. 03_train_teacher: 7/8 main teachers done, botiot_category at epoch 7 (~10 s/epoch); cross and standard teachers still to come. GPU 6.4 GB used, 99% busy. Free space: C: 37.9 GB, D: 622 GB.

### 2026-10-02 03:25-03:30: Slowdown found and fixed (no code or config change)
- All 7 main/cross/standard teachers before it are fine. Teacher test macro-F1 so far (seed 0, strict): ciciot binary 0.945 / family 0.720 /
  class34 0.672; ciciomt binary 0.975 / category 0.816 / attack 0.702; botiot binary 0.977; ciciomt_std attack 0.719.
- botiot_std_category teacher (standard protocol keeps `seq`, so de-dup keeps far more rows): 49,382,244 train rows,
  X_train 6.81 GB (strict botiot: 23.7M rows, 3.18 GB, which fits on the GPU). It exceeds 60% of free VRAM, so "data on device: False"
  and it streamed random rows from the D: HDD at about 15 MB/s, with the GPU about 2% busy and no epoch logged for 10+ min.
- Fix: read botiot_std X_train.npy and y_category_train.npy once, sequentially (55 s), so Windows keeps them in the RAM file cache
  (32 GB RAM, about 15 GB free). Result: D: reads 0 MB/s, GPU 91%, 12.4 s/epoch (strict botiot is about 10 s/epoch).
- PROCEDURE for later wake-ups: if a step logs "data on device: False", or the GPU sits under 10% busy with D: random reads,
  pre-read that dataset's processed X_<split>.npy (plus the y files and teacher cache files it uses) sequentially to warm the cache.
  This is harmless and changes nothing scientific. Candidates: botiot_std (all steps); in Phase 3, parallel GPU jobs may also need it.

### 2026-10-02 03:59: Check: healthy. 03_train_teacher done in 3250 s (54 min). 03b_xgboost running (GPU 94%, D: idle). Free: C: 37.6 GB, D: 621 GB. Results so far:
  - Teacher ciciot_binary_resmlp_s0 test macro-F1 0.9450 acc 0.9914
  - Teacher ciciot_family_resmlp_s0 test macro-F1 0.7200 acc 0.8921
  - Teacher ciciot_class34_resmlp_s0 test macro-F1 0.6719 acc 0.8691
  - Teacher ciciomt_binary_resmlp_s0 test macro-F1 0.9751 acc 0.9959
  - Teacher ciciomt_category_resmlp_s0 test macro-F1 0.8160 acc 0.7907
  - Teacher ciciomt_attack_resmlp_s0 test macro-F1 0.7022 acc 0.7540
  - Teacher botiot_binary_resmlp_s0 test macro-F1 0.9770 acc 1.0000
  - Teacher botiot_category_resmlp_s0 test macro-F1 0.9244 acc 0.9815
  - Teacher xciciot_binary_resmlp_s0 test macro-F1 0.9307 acc 0.9880
  - Teacher xciciot_shared5_resmlp_s0 test macro-F1 0.8096 acc 0.8753
  - Teacher xciciomt_binary_resmlp_s0 test macro-F1 0.9721 acc 0.9951
  - Teacher xciciomt_shared5_resmlp_s0 test macro-F1 0.7723 acc 0.7860
  - Teacher ciciot_std_family_resmlp_s0 test macro-F1 0.7090 acc 0.8818
  - Teacher ciciot_std_class34_resmlp_s0 test macro-F1 0.6850 acc 0.8746
  - Teacher ciciomt_std_category_resmlp_s0 test macro-F1 0.8139 acc 0.7919
  - Teacher ciciomt_std_attack_resmlp_s0 test macro-F1 0.7189 acc 0.7602
  - Teacher botiot_std_category_resmlp_s0 test macro-F1 0.9178 acc 0.9813
  - XGBoost ciciot_binary_s0 test macro-F1 0.9511
  - XGBoost ciciot_family_s0 test macro-F1 0.7293
  - XGBoost ciciot_class34_s0 test macro-F1 0.6856
  - XGBoost ciciomt_binary_s0 test macro-F1 0.9671
  - XGBoost ciciomt_category_s0 test macro-F1 0.7975
  - XGBoost ciciomt_attack_s0 test macro-F1 0.7039
  - XGBoost botiot_binary_s0 test macro-F1 0.9899

### 2026-10-02 04:29: Check: healthy. 03b_xgboost done (2362 s), 03c_baselines done (666 s), 04_shap running (ciciot_binary done, k95=28; IAT is not in its top 5). Free: C: 37.6 GB, D: 621 GB.
- NOTE for the quick-pass report (check e): logreg baseline hit 'lbfgs failed to converge after 300 iterations' 7 time(s) (teacher.yaml baselines.logreg_max_iter=300). The logreg baseline may be slightly under-fit. Report only; config left unchanged (not an allowed tuning).

### 2026-10-02 05:01: Check: healthy. 04_shap running at about 4.5 min per task (main datasets done: ciciot x3, ciciomt x3, botiot x2). seq/IAT not in any top 5 (botiot top 5: spkts, bytes, pkts, proto). GPU 35% (SHAP is CPU-heavy). Free: C: 37.5 GB, D: 621 GB.

### 2026-10-02 05:32: Check: healthy. 04_shap on the standard-protocol models (cross datasets done; 3 left: ciciomt_std x2, botiot_std). Standard-protocol ciciot_std_family has IAT in its top 5, as expected under the leaky protocol. Free: C: 37.5 GB, D: 621 GB.

### 2026-10-02 05:58: Check: healthy. 04_shap done (5020 s), 05_cache_teacher done (397 s, D: free fell by 13 GB to 608 GB). 05_distill running: 77 runs, 3 at a time, 9 done after 7 min. GPU 96%, D: idle (no streaming). Free: C: 37.5 GB.

### 2026-10-02 06:29: Check: healthy but memory-tight
- 05_distill: 46/77 runs done, no errors. GPU 11.2 / 12 GB used, 93% busy (3 jobs on botiot_binary, data on device).
- Host RAM: only 2.4 GB available, commit 49 GB on 32 GB RAM, so the page file (on C:) is in use; C: free 37.5 -> 36.7 GB.
  The 3 distill workers hold about 10.7 / 7.2 / 6.7 GB each (the Bot-IoT runs are the big ones).
- Decision: change nothing yet (no failure; kd.yaml jobs=3 kept). Skipped warming the botiot_std cache: no free RAM to hold it.
- Watch: if a distill worker dies with MemoryError, a CUDA OOM or a pagefile/commit error, lower configs/kd.yaml jobs 3 -> 2
  (allowed OOM fix; finished runs are skipped) and relaunch run_all.py. If C: falls below 8 GB (page file growth), stop and record.

### 2026-10-02 06:45: Check: 52/77 distill runs done, no errors. Bot-IoT category runs are memory-heavy: available RAM 0.1 GB, committed 50.8 GB of a 127.8 GB commit limit (big page file), paging about 50 pages/s, GPU 14% busy, epochs 15-19 s (were 10-14). Slower but not failing; no out-of-memory error, so kd.yaml jobs=3 is unchanged (rule: change only on OOM). Free: C: 36.6 GB, D: 608 GB.

### 2026-10-02 07:06: Check: healthy. 62/77 distill runs done, on to the cross datasets. The Bot-IoT runs finished, RAM pressure eased (4.6 GB available). GPU 87%.
- NOTE for the quick-pass report (check e, int8): xciciot_shared5/kd_all int8 macro-F1 0.2985 vs fp32 0.7814, a big int8 drop. Also botiot_category/kd_shap int8 0.809 vs 0.864. Collect every int8 drop in the report.

### 2026-10-02 07:27: Check: healthy. 70/77 distill runs done (standard-protocol and cross shield runs in progress), no errors. GPU 73%. Free: C: 36.6 GB, D: 608 GB.

### 2026-10-02 07:35: Topology Zoo: upstream fix e9e82e7 pulled; provenance confirmed
- User instruction: pull e9e82e7 (Internet Archive fallback, atomic and validated downloads), test, re-run placement, log provenance.
- git pull (ff, touches only src/shield/placement/topology.py and tests): HEAD e9e82e7. Local config edits untouched.
  `.venv\Scripts\python.exe -m pytest -q`: 42 passed. torch cuda True.
- Placement had NOT run or failed yet (quick pass is still in 05_distill, 73/77). The running run_all.py launches 07_placement.py
  as a new subprocess, so it imports the new topology.py. No relaunch needed and nothing deleted.
- PROVENANCE: the topologies are the Internet Archive copy of topology-zoo.org
  (https://web.archive.org/web/2024id_/http://www.topology-zoo.org/files/<name>.graphml). The files already in
  C:\Users\Prakash\shield\data\topologies (fetched 02:45 from the Zoo authors' GitHub data store mroughan/InternetTopologyZoo)
  were checked byte-for-byte against freshly downloaded Internet Archive copies: all 4 IDENTICAL (SHA-256 match;
  sizes Abilene 9,210 / Geant2012 25,038 / Cogentco 90,926 / Kdl 321,556 bytes). The new download_zoo validates and keeps them.
- Node counts after cleaning (located nodes, largest connected component): Abilene 11, Geant2012 37, Cogentco 180, Kdl 709.

### 2026-10-02 07:56: Check: healthy. 76/77 distill runs done; the last (botiot_std_category/shield) early-stopped at epoch 12 (best val 0.8558) and is in its eval/fidelity stage. No errors. Free: C: 37.2 GB, D: 608 GB. Next: k-sweep, then 06_latency. Keep wake-up checks light while latency runs (CPU must be idle).

### 2026-10-02 08:22: Check: healthy. 05_distill done (7557 s, 77/77). k-sweep running (Bot-IoT k8/k12 in progress). No errors. Free: C: 37.2 GB, D: 608 GB.

### 2026-10-02 08:43: Check: healthy, quick pass near the end
- k-sweep done (2405 s, 32 runs). 06_latency done (43 s; CPU otherwise idle, as no checks ran then).
  Detector throughput 5,981,328 flows/s/core (ciciot_class34 shield), so controller mu = 11,962,656 flows/s.
- 07_placement done (208 s): 2464 runs + 56 brute-force optimum checks, 16 workers, no download problems (cached Archive-identical files).
  Overall average ranks (lower is better): sa 2.35, aco 3.54, hybrid_eho_aco 3.85, ga 4.11, eho 4.45, random 4.75, pso 5.17,
  ilp_ckm 8.81, kmedian 9.37, kmeans 9.81, pagerank 10.36, kcenter 11.45.
  NOTE for check g: the hybrid is 3rd overall; the WARN rule is about the large topologies (Cogentco, Kdl, syn500) and is evaluated in the report.
- 07b_coupled done (58 s). 09_cross_dataset running; 08_make_figures remains.

### 2026-10-02 08:59: QUICK PASS FINISHED. 'Pipeline finished in 379.2 min'. 09_cross_dataset 335 s, 08_make_figures 30 s. No errors. PID 88768 exited. Next: Phase 2 step 3 (QUICK_PASS_REPORT.md), then step 4 (gamma rule).

### 2026-10-02 09:00: PHASE 2 COMPLETE
- QUICK_PASS_REPORT.md written: D:\shield_run\outputs\QUICK_PASS_REPORT.md (analysis script and raw numbers in outputs\quick_pass_analysis\).
  Summary: PASS a, c, d, f, j; WARN b (seq not a strong single-feature leak), e (dtree >= shield on Bot-IoT; large int8 drops;
  kd_variance > shield on 3 tasks), g (hybrid optimality gap mean 2.6%, max 34%; SA best), h (teacher = shield at rho 0.5),
  i (fine-tuned <= scratch in 8 of 32; target norm better in only 3 of 24).
- Step 4 gamma rule: shield best val macro-F1 < kd_shap's on 1 of 8 main tasks (threshold 5), so NOT triggered. gamma stays 0.1; kd.yaml unchanged.
  Decision used validation metrics only.

### 2026-10-02 09:03: PHASE 3 started (full run)
- Restored the full configs: teacher seeds [0,1,2,3,4], kd seeds [0,1,2,3,4], xciciot/xciciomt seeds [0,1,2], placement runs 30.
  Verified: git diff shows configs/teacher.yaml, kd.yaml, datasets.yaml and placement.yaml are identical to HEAD (only paths.yaml differs, local).
  (The first plain-text replace silently missed because of CRLF line endings; redone with regex and verified.)
- Phase 3 GPU check (user amendment): nvidia-smi 740 / 12288 MiB used, so about 11.3 GB free (>= 8 GB): OK. No python.exe was running before launch.
  RAM available 23.3 GB. Disk free C: 37.2 GB, D: 607.7 GB.
- A) GPU chain script D:\shield_run\phase3_gpu_chain.ps1 (fixed two PowerShell pitfalls before launch: flattened single-element arrays
  and the reserved $args variable; syntax-checked). Launched hidden: powershell PID 106212. 03_train_teacher skipped s0 and started
  ciciot_binary s1 (data on device).
- B) 07_placement.py --workers 12: PID 83376. "11200 runs + 0 optimum checks to do (2464 already done)", about 4.5 runs/s, ETA ~45 min.
  It uses the quick-pass latency.json (mu 11,962,656 flows/s).

### 2026-10-02 09:20: Check. B (placement) DONE; A (GPU chain) healthy
- 07_placement.py --workers 12 finished at 09:18:50 (about 15 min): 11200/11200 new runs, runs.jsonl now has 13,664 lines (2464 quick + 11200). No errors.
  Average ranks with 30 seeds: sa 1.77, aco 3.32, hybrid_eho_aco 4.00, eho 4.05, ga 4.51, random 4.74, pso 5.82, ilp_ckm 8.80,
  kmedian 9.35, kmeans 9.81, pagerank 10.36, kcenter 11.48. The hybrid is 3rd overall but nearly tied with eho.
- A: 03_train_teacher on ciciot_family s3 (3.5 s/epoch, GPU 99%). CPU 11%, RAM 20.8 GB available, C: 37.2 GB, D: 607.6 GB. No errors.
- Waiting for A to finish before 07b_coupled.py and 08_make_figures.py.

### 2026-10-02 10:06: Check: A healthy. 03_train_teacher: 27 of about 40 new teachers done (now botiot_binary s4, data on device). No errors. C: 37.2 GB, D: 607.5 GB free.

### 2026-10-02 10:52: Check: A healthy. 03_train_teacher DONE (86.2 min). 03b_xgboost and 03c_baselines had nothing new (xgboost/baselines use seed 0 only). 04_shap running for the new teacher seeds (about 4 min each, about 40 to do, about 3 h). No errors. C: 37.2 GB, D: 607.5 GB free.

### 2026-10-02 11:53: Check: A healthy. 04_shap: 20 new SHAP runs done (through ciciomt_category s4), about 20 left (~1.5 h). No errors. C: 37.2 GB, D: 607.4 GB free.

### 2026-10-02 12:54: Check: A healthy. 04_shap on the cross datasets (xciciot_shared5 s1 next; about 6 left, ~25 min). No errors. RAM 21 GB available. C: 37.2 GB, D: 607.4 GB free. Next: 05_cache_teacher, then the long 05_distill (5 seeds).

### 2026-10-02 13:40: Check: A healthy. 04_shap DONE (165.7 min). 05_cache_teacher DONE (0.1 min; the cache uses teacher_seed 0, already done). 05_distill running: '333 distillation runs, 3 at a time' (includes the 77 seed-0 runs, which are skipped); 87 'macro-F1' lines so far, now ciciot_class34 s3/s4. GPU 92%, RAM 14.7 GB available. No errors. Expect the Bot-IoT runs to be memory-heavy again (see 06:29); rule: lower kd.yaml jobs only on a real OOM.

### 2026-10-02 14:41: Check: A healthy. 05_distill 237 of 333 (on the botiot_binary seeds). GPU 11.5 GB / 76%, RAM 7.2 GB available. No errors. C: 37.2 GB, D: 607.4 GB free.

### 2026-10-02 15:42: Check: A slow but healthy. 05_distill 264 of 333. The botiot_category runs (19 of 35 done) are paging heavily: RAM 0.1 GB available, about 430 pages/s, commit 55.2 of 127.8 GB, epochs 34-54 s (normally ~12 s), GPU 29%. No errors or OOM, so jobs=3 is unchanged (rule: change only on OOM; killing now would lose 3 in-flight runs). Expect ~1.5-2 h until the Bot-IoT runs finish; the remaining cross/std runs are light.

### 2026-10-02 16:43: Check: A healthy. 05_distill 285 of 333; the Bot-IoT runs are done, now on the cross datasets (light). RAM back to 7.7 GB available. No errors. C: 37.0 GB, D: 607.4 GB free.

### 2026-10-02 17:34: Check: A healthy. 05_distill 322 of 333 (xciciomt_shared5, last group). No errors. C: 37.0 GB, D: 607.4 GB free. Next: k-sweep (5 seeds), then 09_cross_dataset.

### 2026-10-02 18:35: Check: A healthy. 05_distill DONE (265.8 min, 333 runs). k-sweep running: 121 of 160 (Bot-IoT runs left, heavier). No errors. C: 37.0 GB, D: 607.4 GB free.

### 2026-10-02 19:36: A (GPU chain) COMPLETE at 19:33:35 (k-sweep 101.6 min, 09_cross_dataset 10.4 min, no errors). Both A and B are done, so running 07b_coupled.py then 08_make_figures.py (Phase 3 step 3).

### 2026-10-02 19:38: PHASE 3 COMPLETE. 07b_coupled: exit 0, 0 new tasks (coupled runs were complete from the quick pass); min controllers unchanged (Geant2012 rho 0.5/0.7/0.9: shield 4/4/4, teacher 4/6/7). 08_make_figures: exit 0, 61 s, 75 tables + 154 figures regenerated. Starting PHASE 4 (final report).

### 2026-10-02 19:50: PHASE 4 COMPLETE, PHASE: DONE
- FINAL_REPORT.md written (beginner-friendly, mean ± std over seeds). Supporting script and numbers: outputs\final_analysis\ (final_analyse.py, print_cross.py, final_results.json).
  Claims checked against the numbers and 7 statements corrected before finishing.
- Key findings: SHIELD = kd_shap statistically (Wilcoxon p=0.64 over 40 seed-task pairs; 20/40 wins); KD CD (p=2.8e-28, CD 1.66):
  scratch_all and kd_all tied best, SHIELD tied with kd_shap/kd_variance/kd_mi. Placement CD (p=3.5e-76, CD 2.23): SA, ACO, hybrid tied best;
  SA best rank (1.77) and gap (0.28%); hybrid gap mean 1.46%, max 15%. Coupling: SHIELD 4 vs teacher 4/6/7 controllers at rho 0.5/0.7/0.9.
- Identical 5-seed mean of shield and kd_shap on ciciot_class34 (55.65) checked per seed: genuinely different runs (gamma 0.1 vs 0.0), coincidence.
- git: nothing to commit or push (no code/config fixes were made; configs restored to HEAD; paths.yaml stays local).
- Total wall-clock: about 17 h (quick pass 6.3 h, full GPU chain 10.5 h, placement 15 min in parallel).
