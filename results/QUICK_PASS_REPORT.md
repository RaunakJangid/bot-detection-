# SHIELD-IoT quick pass report (seed 0)

Run: 2026-10-02 02:29 to 08:59 on the desktop (RTX 3060, 20 logical cores, 32 GB RAM). `Pipeline finished in 379.2 min`, with no errors.
Code: e62c5b0, then 6628d75 (YAML BOM fix) and e9e82e7 (Topology Zoo Internet Archive fallback; placement ran on this code).
Quick-pass configs: teacher/kd seeds [0], cross datasets seeds [0], placement runs 5.
All numbers are **test macro-F1 on seed 0** unless marked "val". One seed gives no error bars, so treat small differences (< 1 point) as noise.
The numbers come from `outputs/data_report_*.json`, `outputs/leakage/`, `outputs/paper/tables/`, `outputs/placement/`, `outputs/coupled/`,
`outputs/cross/` and the per-run `metrics.json` files (analysis script and raw results: `outputs/quick_pass_analysis/`).

## Summary

| Check | Result | One line |
|---|---|---|
| a. Data | **PASS** | Every split built; no main dataset is missing a class in train; de-dup stats reported. |
| b. Leakage audit | **WARN** | IAT is the top single feature on both CIC datasets, but Bot-IoT `seq` is **not** suspicious (rank 25). No other suspects. |
| c. Inflation | **PASS (info)** | Removing IAT/seq changes macro-F1 by at most about 2.3 points (strict vs standard). |
| d. Teachers | **PASS** | All within 3.2 points of XGBoost; none early-stopped at epoch <= 3; IAT/seq absent from strict SHAP. |
| e. KD ablation | **WARN** | dtree beats shield on both Bot-IoT tasks; int8 drops are large; 16-feature students lose 5-7 points on CICIoT family/class34. |
| f. Latency | **PASS** | Student about 130x the teacher's single-core throughput; int8 is about 4x slower than fp32 but 2x smaller. |
| g. Placement | **WARN** | Hybrid is top 3 on Cogentco/Kdl/syn500, but its optimality gap is not near 0% (mean 2.6%, max 34%). |
| h. Coupled | **WARN** | Teacher needs more controllers than shield at rho 0.7 and 0.9, but equal (4 vs 4) at rho 0.5. |
| i. Cross-dataset | **WARN** | Fine-tuned is not better than scratch in 8 of 32 cells; target normalisation helps in only 3 of 24. |
| j. Outputs | **PASS** | Every expected output exists: 75 tables and 154 figures in `outputs/paper/`. |
| Step 4 (gamma) | **Not triggered** | shield's best val macro-F1 < kd_shap's on only 1 of 8 main tasks (threshold 5), so gamma stays 0.1. |

## a. Data: PASS

| Dataset | Protocol | Features (dropped) | Train | Validation | Test | Rows in | Duplicates dropped | Conflicting rows dropped | Rows out |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ciciot | strict | 45 (IAT) | 4,034,235 | 728,213 | 705,401 | 7,845,673 | 2,363,576 | 113,396 (14,248 hashes) | 5,467,849 |
| ciciomt | strict | 44 (IAT) | 3,846,494 | 646,417 | 887,169 | 8,775,013 | 3,391,891 | 859,567 (3,042 hashes) | 5,380,080 |
| botiot | strict | 36 (seq) | 23,699,492 | 5,078,464 | 5,078,459 | 73,370,443 | 39,459,785 | 4,561,131 (54,243 hashes) | 33,856,415 |
| xciciot | strict, shared | 43 | 3,754,948 | 673,562 | 651,282 | 7,396,811 | 2,302,686 | 113,575 | 5,079,792 |
| xciciomt | strict, shared | 43 | 3,622,996 | 606,977 | 823,454 | 8,448,360 | 3,391,891 | 859,567 | 5,053,427 |
| ciciot_std | standard | 46 | 4,114,778 | 740,865 | 716,912 | 7,845,673 | 2,273,118 | 0 | 5,572,555 |
| ciciomt_std | standard | 45 | 3,865,552 | 649,528 | 891,807 | 8,775,013 | 3,368,126 | 0 | 5,406,887 |
| botiot_std | standard | 37 | 49,382,244 | 10,581,910 | 10,581,908 | 73,370,443 | 2,422,639 | 2,015,870 (401,742 hashes) | 70,546,062 |

- No dataset is missing a class in its training split.
- **Note on the CICIoT2023 source.** This machine's CICIoT zip (`archive (1).zip`, 2.3 GB uncompressed, 3 CSVs) gives 7.85 M rows before de-dup.
  The full CICIoT2023 release is about 46 M rows, so this is a **subset**, and the paper should describe it as such.
- Bot-IoT: removing `seq` makes 54% of rows exact duplicates of another row (39.5 M of 73.4 M), so strict Bot-IoT is less than half the size of the standard set.
  The duplication comes from the recording schedule, not new information. This is worth one sentence in the paper.

## b. Leakage audit: WARN

| Audit task | All-features macro-F1 | Chance | Leaky feature | Its rank | Its macro-F1 alone | Other features >= 80% of all-features |
|---|---:|---:|---|---:|---:|---|
| ciciot_std / class34 | 0.475 | 0.029 | IAT | **1** | **0.530** (higher than all features together) | none |
| ciciomt_std / attack | 0.575 | 0.053 | IAT | **1** | **0.654** (higher than all features together) | none |
| botiot_std / category | 0.779 | 0.200 | seq | **25** | 0.153 | none |

- IAT is a strong single-feature leak on both CIC datasets: on its own it beats the model that uses every feature.
- **WARN:** in this audit `seq` does not look suspicious. Bot-IoT's top single features are sbytes 0.608, dur 0.486, spkts 0.472, bytes 0.458 and rate 0.447.
  The strict protocol still removes `seq` (protocol unchanged). The paper's argument for removing `seq` must rest on its meaning
  (an Argus record counter) and on the de-duplication effect, not on this audit.

## c. Inflation (strict vs standard protocol): PASS (info)

| Task | Model | Strict | Standard | Inflation (points) |
|---|---|---:|---:|---:|
| ciciot family | teacher / shield | 72.00 / 61.97 | 70.90 / 61.55 | -1.10 / -0.42 |
| ciciot class34 | teacher / shield | 67.19 / 55.58 | 68.50 / 57.61 | +1.31 / +2.03 |
| ciciomt category | teacher / shield | 81.60 / 79.75 | 81.39 / 79.33 | -0.21 / -0.42 |
| ciciomt attack | teacher / shield | 70.22 / 64.49 | 71.89 / 66.81 | +1.67 / +2.32 |
| botiot category | teacher / shield | 92.44 / 85.06 | 91.78 / 85.64 | -0.65 / +0.58 |

The leak inflates the fine-grained tasks (class34, attack) by about 1.3-2.3 points. On the coarse tasks the difference is within single-seed noise.
This is much smaller than the single-feature audit suggests: the multi-feature models mostly recover the same information from other features.

## d. Teachers: PASS

| Task | Teacher | XGBoost | Teacher - XGB | Last / best epoch |
|---|---:|---:|---:|---|
| ciciot binary | 94.50 | 95.11 | -0.61 | 37 / 31 |
| ciciot family | 72.00 | 72.93 | -0.94 | 30 / 24 |
| ciciot class34 | 67.19 | 68.56 | -1.38 | 18 / 12 |
| ciciomt binary | 97.51 | 96.71 | +0.80 | 40 / 40 |
| ciciomt category | 81.60 | 79.75 | +1.85 | 40 / 35 |
| ciciomt attack | 70.22 | 70.39 | -0.17 | 40 / 38 |
| botiot binary | 97.70 | 98.99 | -1.29 | 11 / 5 |
| botiot category | 92.44 | 95.63 | **-3.19** | 19 / 13 |

- No teacher is more than 5 points below XGBoost, and none early-stopped at epoch <= 3.
- IAT/seq are not in any strict teacher's SHAP importance.
- Info: the three CICIoMT teachers hit the 40-epoch cap with their best epoch at 35-40, so they may still be improving. That is not a WARN under the rules.
  Bot-IoT category is the largest gap to XGBoost (-3.2).

## e. KD ablation: WARN

Test macro-F1 (%), seed 0, k = 16 for the selection variants:

| Task | teacher | xgboost | dtree | logreg | scratch_all | kd_all | kd_shap | **shield** | shield int8 | kd_random | kd_mi | kd_variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ciciot binary | 94.50 | 95.11 | 92.59 | 89.16 | 94.56 | 94.11 | 93.77 | **93.82** | 93.05 | 92.76 | 92.87 | 93.74 |
| ciciot family | 72.00 | 72.93 | 59.59 | 59.17 | 67.55 | 66.35 | 61.93 | **61.97** | 56.76 | 61.32 | 60.31 | 60.83 |
| ciciot class34 | 67.19 | 68.56 | 50.01 | 55.50 | 62.89 | 60.99 | 55.26 | **55.58** | 51.17 | 54.25 | 55.34 | 50.44 |
| ciciomt binary | 97.51 | 96.71 | 94.97 | 95.43 | 97.12 | 97.56 | 96.69 | **97.28** | 97.28 | 94.96 | 95.35 | 96.02 |
| ciciomt category | 81.60 | 79.75 | 78.37 | 75.54 | 79.90 | 81.97 | 78.07 | **79.75** | 70.54 | 77.25 | 78.84 | 82.53 |
| ciciomt attack | 70.22 | 70.39 | 61.36 | 60.11 | 66.76 | 67.84 | 64.54 | **64.49** | 44.89 | 58.37 | 64.85 | 65.05 |
| botiot binary | 97.70 | 98.99 | **98.12** | 86.02 | 97.36 | 97.51 | 97.12 | **97.90** | 90.07 | 96.28 | 97.27 | 97.12 |
| botiot category | 92.44 | 95.63 | **85.72** | 65.74 | 92.17 | 90.55 | 86.44 | **85.06** | 79.56 | 86.57 | 85.84 | 89.56 |

- **WARN (dtree within 1 point of shield):** on Bot-IoT, the 10-level decision tree is *better* than shield, by 0.22 on binary and 0.66 on category.
  logreg is never within 1 point. (logreg also failed to converge within 300 iterations on 7 fits, so it may be slightly under-fit.)
- **shield vs kd_shap** (same 16 features; shield adds the attribution loss): shield > kd_shap on 6 of 8 tasks. The gains are small (+0.05 to +0.3 on CICIoT,
  +0.6 on ciciomt binary, +1.7 on ciciomt category, +0.8 on botiot binary); the losses are -0.05 on ciciomt attack and -1.4 on botiot category.
  With one seed this is mostly within noise. Phase 3's 5 seeds will tell.
- **shield vs the other 16-feature selectors** (random/mi/variance): shield is best on ciciot binary/family/class34, ciciomt binary and botiot binary,
  but **kd_variance beats shield** on ciciomt category (82.53 vs 79.75), ciciomt attack (65.05 vs 64.49) and botiot category (89.56 vs 85.06),
  and kd_mi also edges it on ciciomt attack (64.85).
- **Cost of keeping 16 features:** on CICIoT family/class34, the all-feature students (scratch_all/kd_all) are 4-7 points above shield.
- **Parameters:** shield has 3.2-4.3 K parameters vs the teacher's 1.20-1.22 M (about 300x fewer).
- **SHAP fidelity** (Spearman between student and teacher rankings on the student's features): shield 0.28-0.87; it is higher than kd_shap
  on the CICIoT tasks (0.54-0.64 vs 0.36-0.52), similar elsewhere, and lowest on botiot category (0.28). kd_all is 0.75-0.96, but over all features, so it is not directly comparable.
- **WARN, int8 quantisation drop:** int8 is often far worse than fp32. For shield: ciciomt attack 64.49 -> 44.89, ciciomt category 79.75 -> 70.54,
  botiot binary 97.90 -> 90.07, botiot category 85.06 -> 79.56. The all-feature students collapse even more (ciciot binary scratch_all 94.56 -> 38.23,
  ciciot class34 kd_all 60.99 -> 25.59). SHAP-selected students hold up best. This looks systematic. It may be a quantisation design limitation
  (for example per-tensor scaling on inputs with very different ranges), not something to tune here. **For the user to decide:** whether to present int8 at all, or investigate.

## f. Latency (single core, batch 256, flows/s): PASS

| Task | Teacher | shield (fp32) | shield int8 | shield / teacher | int8 / fp32 | Size fp32 -> int8 (KB) |
|---|---:|---:|---:|---:|---:|---|
| ciciot binary | 48,703 | 6,564,131 | 1,462,023 | 135x | 0.22 | 15.3 -> 8.1 |
| ciciot family | 48,411 | 6,614,996 | 1,500,587 | 137x | 0.23 | 16.1 -> 8.3 |
| ciciot class34 | 48,197 | 5,981,327 | 1,490,972 | 124x | 0.25 | 19.5 -> 9.3 |
| ciciomt binary | 49,004 | 6,513,964 | 1,491,842 | 133x | 0.23 | 15.3 -> 8.1 |
| ciciomt category | 48,653 | 6,481,020 | 1,488,372 | 133x | 0.23 | 15.8 -> 8.3 |
| ciciomt attack | 48,280 | 6,243,890 | 1,485,780 | 129x | 0.24 | 17.5 -> 8.7 |
| botiot binary | 48,942 | 6,580,946 | 1,503,229 | 134x | 0.23 | 15.3 -> 8.1 |
| botiot category | 48,867 | 6,514,003 | 1,496,637 | 133x | 0.23 | 15.7 -> 8.2 |

- The student is 124-137x faster than the teacher on one core. The CPU was otherwise idle (06_latency ran with no other pipeline step, and no checks during it).
- As on the laptop, int8 is about 4x **slower** than fp32 and about 2x smaller. Present it as "smaller, not faster".
- Placement uses the ciciot_class34 shield throughput: 5,981,328 flows/s/core, giving controller mu = 11,962,656 flows/s.
- This `latency.json` is the source of truth for Phase 3 (06_latency is not re-run while placement runs).

## g. Placement: WARN

- Friedman test over all instances: p = 8.1e-74. Average ranks (lower is better): sa 2.35, aco 3.54, **hybrid_eho_aco 3.85**, ga 4.11, eho 4.45,
  random 4.75, pso 5.17, ilp_ckm 8.81, kmedian 9.37, kmeans 9.81, pagerank 10.36, kcenter 11.45.
- **Large topologies (mean rank across the k/rho settings):** Cogentco: sa 2.00, **hybrid 2.50 (2nd)**; Kdl: sa 1.43, aco 2.57, **hybrid 2.79 (3rd)**;
  syn500: sa 1.57, aco 2.50, **hybrid 3.36 (3rd)**. The hybrid is within the top 3 on all three, so the large-topology rule passes.
- **WARN, optimality gap vs brute force (small instances, 56 checks):** the hybrid's gap is mean 2.65%, median 0.11%, max 34.3% (Geant2012 k=4 rho=0.7).
  It hits 0% on only 45% of instances. For comparison: sa mean 0.20% (max 1.2%), random search 0.66%, eho 0.68%, aco 1.12%. So the hybrid is **not** "at or near 0%".
- Runtime per run (mean): hybrid 0.53 s, sa 0.52 s, eho 0.47 s, ga 0.67 s, aco 2.28 s, pso 2.78 s, ilp_ckm 9.35 s.
- Same budget for every metaheuristic (20,000 evaluations). With this budget SA is the strongest method. That undercuts the "hybrid is best" claim;
  the paper may need to frame the hybrid differently, or the user may want to revisit its settings (not done here: not an allowed change).

## h. Coupled (controllers needed, Geant2012): WARN

| rho | shield | shield int8 | teacher |
|---|---:|---:|---:|
| 0.5 | 4 | 4 | **4** |
| 0.7 | 4 | 4 | 6 |
| 0.9 | 4 | 4 | 7 |

The teacher needs more controllers at rho 0.7 and 0.9 (+2 and +3), but not at rho 0.5 (tie), so "more at every load level" does not hold.
Only Geant2012 is in the coupled experiment's output.

## i. Cross-dataset (xciciot <-> xciciomt, shared features/classes): WARN

- **Zero-shot drop** (in-domain -> other dataset, source normalisation): binary drops 0.14-0.32 (xciciomt->xciciot) and 0.17-0.39 (xciciot->xciciomt);
  shared5 drops 0.18-0.47. The best zero-shot model is mixed: shield is best on xciciomt->xciciot binary with target norm (0.867) and on
  xciciot->xciciomt shared5 with source norm (0.577); XGBoost and kd_all win elsewhere.
- **Target vs source normalisation:** target normalisation is better in only **3 of 24** model/direction/task cells (shield and teacher on
  xciciomt->xciciot binary; teacher on xciciomt->xciciot shared5). Usually source normalisation wins.
- **Few-shot (fine-tuned vs scratch):** **WARN:** fine-tuned is not better than scratch in 8 of 32 cells, mostly at the smallest
  fraction (0.1%) and for xciciomt->xciciot binary shield (0.1%, 1%, 5%). Fine-tuning helps most on shared5 at 0.1-1% (up to +15 points for shield).
- **Never-seen attack families** (binary detector, detection rate): xciciomt->xciciot: Mirai is detected well by everything (0.67-1.00);
  BruteForce and Web are poorly detected (shield 0.05-0.13 and 0.10-0.12; XGBoost with target norm best, 0.76 / 0.85).
  xciciot->xciciomt: MQTT attacks 0.98-1.00 for every model.
- **SHAP agreement between the datasets:** binary Spearman 0.745, top-10 overlap 0.8; shared5 Spearman 0.876, top-10 overlap 0.8.

## j. Paper outputs: PASS

`run_all.expected()` patterns: none missing. `outputs/paper/tables/`: 75 files (CSV + LaTeX); `outputs/paper/figures/`: 154 files.

## Step 4: gamma tuning rule: not triggered

The rule compares shield's best **validation** macro-F1 with kd_shap's (max over `history[].val_macro_f1`, seed 0, 8 main tasks):
shield is below kd_shap on **1 of 8** tasks (threshold 5). So no sweep: `loss.gamma` stays 0.1 and kd.yaml is unchanged. No test data was used for this decision.

## WARNs for the user to decide on later (nothing was changed)

1. Bot-IoT `seq` is not a strong single-feature leak in the audit (b). The justification for removing it needs to rest on what `seq` means.
2. On Bot-IoT, a decision tree beats shield, and kd_variance beats shield on ciciomt category/attack and botiot category (e).
3. int8 quantisation loses a lot of accuracy (up to 20 points for shield, more for the all-feature students) and is slower than fp32 (e, f).
4. The hybrid EHO-ACO is not near-optimal on small instances; SA is better on gap and rank (g).
5. Coupling: teacher and shield tie at rho 0.5 (h).
6. Cross-dataset: fine-tuning does not always beat scratch; target normalisation rarely helps (i).
7. Data scope: the CICIoT2023 input is a subset (7.85 M of about 46 M rows) (a).

