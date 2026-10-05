# SHIELD-IoT v2 unattended run: state log

Instructions: D:\shield_run\outputs_v2\RUN_INSTRUCTIONS.md (the v2 prompt plus all v1 rules). Re-read it on every wake-up.
v1 results (never touch): D:\shield_run\outputs\ (FINAL_REPORT.md, RUN_STATE.md).

## Current
PHASE: DONE (2026-10-04 ~23:00). Report: D:\shield_run\outputs_v2\FINAL_REPORT_v2.md. Loop stopped.
OPEN ITEM FOR USER: git push needs an interactive GitHub login on this desktop. The 2 results commits were rebased onto the rewritten origin/main (eb6a9ba) on 2026-10-05: a39f1ea (v1 results) and ca5bb42 (v2 results). Ready to push: run 'git push origin main' from C:\Users\Prakash\shield (plain push, no force needed).
Local-only changes NOT committed: configs/paths.yaml (machine paths), configs/kd.yaml jobs 3->2 (this machine's RAM).

## Log (append only)
### 2026-10-04 ~00:55: V2 Phase 1 (preflight)
- git pull --rebase --autostash: HEAD e1f85e4 (v2), with my unpushed v1-results commit rebased on top (a944c40). Local configs/paths.yaml kept.
- pytest (.venv\Scripts\python.exe -m pytest -q): 56 passed, 1 FAILED, not the expected 57:
  tests/test_placement.py::test_hybrid_finds_brute_force_optimum: hybrid hits the brute-force optimum on 28/30 seeds (needs >= 29).
  The two misses (seeds 1 and 26) are 0.0091% above the optimum. Deterministic on this machine (numpy 2.5.3, Intel CPU), so it is a borderline
  quality threshold, not a crash or bug. DECISION: no code or test change (not an allowed fix); continue and record it as a WARN for the user.
- torch cuda True.
- configs/paths.yaml: outputs_dir "D:/shield_run/outputs_v2" (data_dir unchanged). No BOM; Paths() loads D:\shield_run\data and D:\shield_run\outputs_v2.
- v2 commit touches no ingest/preprocess code, so reusing the processed data in D:\shield_run\data is valid.
- Smoke test: run with --smoke --force. --force only affects data_smoke/outputs_smoke (disposable), so the smoke tests v2 code
  end to end instead of skipping steps on stale v1 smoke artifacts. The real run will not use --force.

### 2026-10-04 00:45: V2 Phase 1 complete; full run launched
- Smoke test (v2 code, --smoke --force): PASSED. 18 steps in 5.8 min, "Smoke check passed: every expected output exists."
  New stages ran: 05a_tune_kd 22 s, 05_distill --low-data 24 s, 07a_tune_placement 4 s. Only noise: logreg ConvergenceWarning (smoke caps max_iter at 50).
- GPU check: 963 / 12288 MiB used, so about 11.3 GB free (>= 8 GB): OK. No python.exe running before launch. RAM 19.5 GB available. Free: C: 38.5 GB, D: 607.4 GB.
- Launched: Start-Process .venv\Scripts\python.exe scripts\run_all.py (no --smoke, no --force), PID 41356, 2026-10-04 00:45:58.
  After 60 s: ingest/preprocess reused the v1 data; 02b leakage audit running. No errors.
- No experiment config was edited (kd/teacher/placement/datasets/cross/coupled.yaml are at e1f85e4 as committed).

### 2026-10-04 01:08: Check: healthy. 02b leakage audit done (169 s). 03_train_teacher running all 5 seeds (now ciciot_family s3, 3.5 s/epoch, GPU 99%). No errors. RAM 17.4 GB available. C: 38.5 GB, D: 607.3 GB free. Watch for botiot_std_category (49 M rows, 'data on device: False') and warm the cache if it streams slowly (v1 procedure).

### 2026-10-04 01:39: Check: healthy. 03_train_teacher: 20 of about 57 teachers done (fresh outputs_v2, so seed 0 is retrained with the new 80-epoch cap); now ciciomt_category s0. No errors. GPU 99%. C: 38.5 GB, D: 607.3 GB free.

### 2026-10-04 02:25: Check: healthy. 03_train_teacher 39 of 57 (botiot_category s4, 10 s/epoch, data on device). No errors. Pre-read botiot_std X_train.npy + y_category_train.npy (6.9 GB) into the RAM cache ahead of the botiot_std teacher (v1 streaming fix, allowed) in 53 s. C: 38.1 GB, D: 607.2 GB free.

### 2026-10-04 03:07: Check: healthy. 03_train_teacher DONE (7936 s, 57 teachers). botiot_std teacher (data on device: False) started 02:56 and finished before 03:02, so the cache pre-read worked (no stall; v1 stalled for 10+ min). 03b_xgboost running (ciciot binary 0.9511, family 0.7293, identical to v1, same seed and settings). No errors. C: 38.4 GB, D: 606.9 GB free.

### 2026-10-04 03:53: Check: healthy. 03b_xgboost DONE (2361 s), 03c_baselines DONE (661 s). 04_shap (class-balanced in v2) just started; in v1 SHAP over all seeds took about 4 h. No errors. C: 38.3 GB, D: 605.9 GB free.

### 2026-10-04 04:54: Check: healthy. 04_shap 13 of 57 in about 60 min (about 4.4 h total, ETA ~08:15). No errors. C: 38.3 GB, D: 605.9 GB free.

### 2026-10-04 05:55: Check: healthy. 04_shap 27 of 57. No errors. C: 38.3 GB, D: 605.9 GB free.

### 2026-10-04 06:56: Check: healthy. 04_shap 40 of 57 (ETA ~08:15). No errors. C: 38.3 GB, D: 605.8 GB free. Next: 05_cache_teacher (v2 cache format), 05a_tune_kd, 05_distill.

### 2026-10-04 07:57: Check: healthy. 04_shap 53 of 57 (the standard-protocol models are last). No errors. C: 38.3 GB, D: 605.8 GB free.

### 2026-10-04 08:28: Check: healthy. 04_shap DONE (15755 s = 4.4 h). 05_cache_teacher DONE (507 s; D: free 605.8 -> 592.8 GB). 05a_tune_kd running: 96 tuning runs (seed 0, validation only), 3 at a time, about 7 done in 6 min (ETA ~1.5 h). GPU 10.6 GB / 97%, RAM 8.9 GB available. No errors.

### 2026-10-04 09:29: Check: healthy. 05a_tune_kd still running (now the DKD grid on Bot-IoT). RAM 8.6 GB available, no paging, GPU 78%. No errors. C: 38.3 GB, D: 592.8 GB free.

### 2026-10-04 10:15: Check: healthy. 05a_tune_kd (validation only): Step 1 objective = {'kd': 'kd', 'T': 1.0, 'alpha': 0.7, 'beta': 0.3} (mean val 0.8445; DKD not chosen). Step 2 teacher = teacher (not the assistant). Step 3 (per-task k/student size): 48 of 160 runs, ETA ~1.5 h. RAM 6.0 GB available, no paging. No errors. C: 38.3 GB, D: 586.8 GB free.

### 2026-10-04 11:16: Check: healthy. 05a_tune_kd step 3: 87 of 160 (the Bot-IoT size grid is slower). RAM 8.1 GB available, light paging (~49 pages/s). No errors. C: 38.3 GB, D: 586.8 GB free.

### 2026-10-04 12:17: Check: healthy. 05a_tune_kd step 3: 122 of 160. RAM 12.1 GB available. No errors. C: 38.3 GB, D: 586.8 GB free.

### 2026-10-04 13:08: Check: healthy. 05a_tune_kd step 3: 154 of 160 (last Bot-IoT size runs). RAM 11 GB available. No errors. C: 38.3 GB, D: 586.8 GB free.

### 2026-10-04 13:54: Check: healthy. 05a_tune_kd DONE (17651 s = 4.9 h). Step 3 ranking = class_balanced; per task (k, hidden): ciciot_binary 16 [64,32]; ciciot_family 32 [128,64]; ciciot_class34 32 [128,64]; ciciomt_binary 12 [64,32]; ciciomt_category 16 [128,64]; ciciomt_attack 32 [128,64]; botiot_binary 12 [64,32]; botiot_category 12 [128,64]. 05_distill running: 333 runs, 3 at a time, 72 done in 37 min. RAM 11.3 GB available. No errors. C: 38.3 GB, D: 586.7 GB free.

### 2026-10-04 14:55: Check: healthy. 05_distill 194 of 333 (CICIoMT attack; Bot-IoT next). RAM 12.5 GB available, no paging. No errors. C: 38.3 GB, D: 586.7 GB free.

### 2026-10-04 15:46-15:48: Heavy RAM paging, so kd.yaml jobs lowered 3 -> 2 and relaunched (allowed engineering fix)
- 05_distill at 224 of 333, now on the Bot-IoT runs: RAM 0.1 GB available, commit 50.9 GB (32 GB RAM), epochs 23-36 s
  (the same runs take ~10-13 s without memory pressure), GPU only 38% busy. Same pattern as v1, where it cost hours.
- Checked first: 05a_tune_kd saves outputs/tuning/kd_choice.json and skips when it exists (it does), so the 4.9 h tuning is NOT redone.
  Other finished steps skip their existing outputs.
- 15:47: stopped PIDs 41356 (run_all), 141216, 147400, 147472 (05_distill) and 3 multiprocessing workers. The 3 in-flight Bot-IoT runs are
  lost (they restart; finished runs are kept). Confirmed no python.exe left.
- configs/kd.yaml: `jobs: 3` -> `jobs: 2` (one line; written BOM-free; load_config shows jobs 2, smoke jobs 1). This is the
  only experiment-config edit, explicitly allowed by the v2 instructions. It does not change any result, only concurrency.
- 15:48:13 relaunched: Start-Process .venv\Scripts\python.exe scripts\run_all.py (no --force), PID 229264, log logs\run_v2_b.log.
  01/02 skipped; 02b_leakage_audit reruns (no skip check; deterministic, about 3 min, same outputs).

### 2026-10-04 16:12: Check: relaunch healthy. Earlier steps skipped (tuning: 'KD tuning already done'); 05_distill '333 runs, 2 at a time' resumed and is on Bot-IoT category (botiot_binary finished). GPU 92% (was 38%), RAM 9.5 GB available, ~2 pages/s (was RAM 0.1 GB). No errors.

### 2026-10-04 17:13: Check: healthy. 05_distill ~315 of 333 (the Bot-IoT runs finished; now the cross datasets). RAM 14.6 GB available, no paging. No errors. C: 38.0 GB, D: 586.7 GB free. Next: k-sweep, low-data, latency, tune placement, placement, coupled, cross, figures.

### 2026-10-04 18:14: Check: healthy. 05_distill DONE (relaunch part 6007 s). k-sweep running: 160 runs, 2 at a time, 44 done in 43 min (Bot-IoT runs still to come, ETA ~2.5-3 h). RAM 15.7 GB available. No errors.

### 2026-10-04 19:15: Check: healthy. k-sweep 119 of 160. Pages/sec ~40-50k, but page reads only ~200/s and RAM 10-13 GB available, so this is memory-mapped data being read (training .npy), not RAM pressure. Epochs at normal speed. No action. No errors.

### 2026-10-04 20:16: Check: healthy. k-sweep 152 of 160 (last Bot-IoT category runs). RAM 7.7 GB available. No errors. Next: 05_distill --low-data.

### 2026-10-04 21:07: Check: healthy. k-sweep DONE (10872 s). 05_distill --low-data running: 240 runs, 2 at a time, 180 done in 35 min (small fractions are fast). RAM 13.5 GB available. No errors. Next: 06_latency (CPU must be idle: no heavy checks during it), 07a_tune_placement, 07_placement, 07b_coupled, 09_cross, 08_figures.

### 2026-10-04 21:48: Check (light): healthy. low-data DONE; 06_latency DONE (42 s; no checks ran during it, CPU idle apart from the pipeline); 07a_tune_placement DONE (131 s); 07_placement running 7500/18704 runs (~20 min left). No errors, no download problems.

### 2026-10-04 22:19: Check: healthy. 07_placement DONE (1305 s, 18,704 runs). 07b_coupled DONE (66 s). 09_cross_dataset running (now with CORAL and DANN; last pair xciciomt->xciciot shared5). Then 08_make_figures. No errors.

### 2026-10-04 22:35: V2 PIPELINE FINISHED. 'Pipeline finished in 392.6 min' (relaunched part; first launch ran 00:46-15:47). 09_cross 1341 s, 08_figures 28 s. No errors. 87 tables, 158 figures. Next: FINAL_REPORT_v2.md, then copy to repo results\v2\ and push.

### 2026-10-04 ~22:35-23:00: V2 report, copy, commit, push attempt; DONE
- FINAL_REPORT_v2.md written (items A-H, both CD diagrams, limitations, v1 comparison). Scripts and numbers in outputs_v2\final_analysis\.
  Claims checked against the data (one statement corrected: kd_all > scratch_all on 7 of 8 tasks, not 6).
- Key v2 findings: student-teacher gap on CICIoT family/34-class 10.2/11.8 -> 2.4/3.7 points (bigger tuned students: k=32, 128-64);
  tuning chose plain KD T=1 alpha=0.7 and the teacher (no TA; DKD not chosen); KD now matches scratch on the hard tasks (all within ±);
  SHIELD vs kd_shap still no difference (p=0.10, -0.15 points); low-data Wilcoxon non-significant at 0.1/1/10%;
  CORAL beats the old normalisations in 16/24, DANN in 7/8 (SHIELD CICIoT->CICIoMT binary 0.61 -> 0.86);
  placement: memetic hybrid ranked 1st (3.44), tied group with no-ants/SA/PSO/ACO/GA, mean gap 0.46% (v1 1.46%), optimum on 82% of instances,
  ants add nothing (no-ants ties), local search is the key; vs SA: significantly better on 8/56, worse on 0;
  coupling: SHIELD saves 1-4 controllers on Geant2012, 2 (Cogentco) and 1 (syn200) only at load 0.9; int8 drops up to 22 points on the larger students.
- Copied FINAL_REPORT_v2.md + 43 CSVs (0.36 MB) to repo results\v2\. Commit f6ebad5 "Add v2 results: final report and paper tables" (on top of a944c40, v1 results).
- git push FAILED: no GitHub credentials on this machine ("could not read Username ... terminal prompts disabled"). Credential helper is Git
  Credential Manager, which needs an interactive browser login. Left for the user. Nothing else was pushed.
- Total v2 wall-clock about 22 h (00:46 -> 22:33, including the 15:47 restart).

### 2026-10-05: Results commits made push-able onto the rewritten GitHub history (user instruction; no push by me)
- git stash push configs/kd.yaml configs/paths.yaml (local-only changes).
- git fetch origin: origin/main force-updated e1f85e4 -> eb6a9ba ("Hybrid optimum test: require <=0.1% gap on 95% of seeds"), on top of the
  rewritten c92722c (v2) and 36e26f5 (Topology Zoo). Fetching the public repo needed no sign-in.
- merge-base: e1f85e4 is an ancestor of HEAD, not of origin/main, so the commits sat on the old history:
  git rebase --onto origin/main e1f85e4: clean, 2/2 commits.
- Both commit messages had a "Co-Authored-By: Claude ..." line; removed from these 2 commits only:
  git filter-branch -f --msg-filter "sed '/^Co-Authored-By/d' | git stripspace" origin/main..HEAD (needs Git's usr\bin on PATH).
  Result: a39f1ea "Add full-run results: reports, paper tables and figures", ca5bb42 "Add v2 results: final report and paper tables".
  No Co-Authored-By / Claude / Anthropic text left in either message.
  (Note: the v2 message body still mentions "pipeline v2 (e1f85e4)", the pre-rewrite hash; the same code is now c92722c. Left unchanged, as only the attribution was to be removed.)
- git diff --stat origin/main..HEAD: 281 files, 12,139 insertions, all under results/ (no other paths).
- git stash pop: kd.yaml (jobs: 2) and paths.yaml (outputs_v2) restored, still uncommitted. Stash list empty.
- pytest (.venv\Scripts\python.exe -m pytest -q): 57 passed (the updated hybrid test now passes on this machine). torch cuda True.
- Branch state: main ahead of origin/main by 2 (a39f1ea, ca5bb42). Not pushed (left to the user, who signs in interactively).
- filter-branch leaves a backup ref refs/original/refs/heads/main (harmless; it is never pushed by 'git push origin main').

### 2026-10-05: PUSHED by the user (interactive sign-in): eb6a9ba..ca5bb42 main -> main. Verified with fetch: origin/main = ca5bb42, local main in sync, 281 results/ files on GitHub. Open item closed.

### 2026-10-05 ~00:50: Two-sided placement statistics (user instruction, commit 82bc94a); analysis only, no new experiments
- git pull --rebase --autostash: HEAD 82bc94a ("Placement statistics: test both directions"); local kd.yaml/paths.yaml kept uncommitted.
- Backed up the old Wilcoxon tables to outputs_v2\logs\pre_82bc94a\.
- 07_placement.py --analyse-only (exit 0, ranks unchanged) and 08_make_figures.py (exit 0). Logs: logs\p_analyse_82bc94a.log, logs\figs_82bc94a.log.
- New placement_wilcoxon.csv: hybrid vs SA significantly better 11 / worse 2 of 56 (Geant2012 k=4 and k=6, rho=0.5), Holm-corrected each direction;
  mean_diff_pct +0.33 (hybrid minus SA, lower is better), so SA is slightly better on average (driven by Cogentco k=8 rho=0.5 +6.8%, Geant2012 k=4 rho=0.7 +6.6%).
  Per instance: hybrid lower mean on 27, SA on 10, equal on 19. Vs others: worse on at most 2 (ACO 2, EHO 1, PSO 1, rest 0).
- FINAL_REPORT_v2.md section F.2 updated, with a correction note that the old "better on 8, worse on 0" came from the one-sided test. Mirrored to D:\shield_run\outputs_v2.
- results\v2\placement_wilcoxon.csv replaced. Commit 68cedbd "v2 report: two-sided placement statistics vs SA" (no AI attribution;
  the first attempt failed because PS 5.1 split the message at embedded quotes, so git commit -F <file> was used instead).
- Pushed with the stored GitHub login: 82bc94a..68cedbd main -> main. Verified local main = origin/main.
