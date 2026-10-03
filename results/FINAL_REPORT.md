# SHIELD-IoT: final results report

*Written for you: someone new to machine learning and networking. Each section starts with what it means in plain words, followed by the numbers.*

**Run:** 2 October 2026, 02:24 to 19:38 (about 17 hours wall-clock) on the desktop (RTX 3060 12 GB, 20 CPU threads, 32 GB RAM).
**Code:** commit e9e82e7 (plus the earlier fixes 6628d75 and e62c5b0). No code or config was changed by me during the run.
**Seeds:** teachers and students 5 seeds (0-4), cross-dataset models 3 seeds (0-2), placement 30 runs per setting. Unless marked otherwise,
numbers are **test macro-F1 in %, shown as mean ± standard deviation over seeds**. XGBoost, the decision tree and logistic regression ran with 1 seed, so they have no ±.

> **Two words used everywhere.**
> *Macro-F1* is a score from 0 to 100 that treats every attack class as equally important, so a model cannot look good by only getting the common classes right.
> *Standard deviation (±)* tells you how much the result moved between repeated runs with different random seeds. If two methods differ by less than their ± values, treat them as the same.

---

## 1. The short version

1. **Small, fast detectors work.** The distilled "student" models have about 300x fewer parameters than the "teacher" and run about **130x faster** on one CPU core
   (6.0-6.6 million flows/s vs about 48,000), while staying within 0.1-0.8 points of the teacher on the binary tasks, and 0.5-12 points below it on the
   multi-class tasks (worst: CICIoT family -10.2 and 34-class -11.8).
2. **But the SHIELD-specific trick (the SHAP attribution loss) did not improve accuracy.** SHIELD and the same model without that extra loss (`kd_shap`) are statistically
   indistinguishable (paired Wilcoxon test over 40 seed-task pairs: p = 0.64; SHIELD wins exactly 20 of 40). It *does* make the student's feature
   ranking agree more with the teacher's on CICIoT (fidelity 0.54-0.64 vs 0.36-0.52), but not on the other datasets (lower on CICIoMT binary/category).
3. **Keeping only 16 features costs accuracy on the hard tasks.** Students that use all 45 features (`scratch_all`, `kd_all`) rank best overall and beat
   SHIELD by 5.7-7.3 points on CICIoT family and 34-class.
4. **Leakage is real but small once you use many features.** The suspicious feature IAT alone beats the full model on both CIC datasets, but removing it only lowers
   fine-grained scores by about 1.3-2.3 points. Bot-IoT's `seq` turned out **not** to be a strong leak in the audit.
5. **Models trained on one dataset do much worse on the other** (cross-dataset), with drops of 11-47 points. A little labelled data from the new dataset (0.1-1%)
   recovers a lot for the 5-class task (up to +17 points for SHIELD).
6. **Controller placement: the hybrid EHO-ACO is good but not the best.** Simulated annealing (SA) is best on average rank and on gap to the true optimum.
   Statistically, the hybrid is tied with SA and ACO, but it is not near-optimal on small networks (mean gap 1.5%, worst 15%).
7. **Coupling works for medium and high load.** Because SHIELD is so fast, the network needs 4 controllers where the teacher needs 6 (load 0.7) or 7 (load 0.9).
   At load 0.5 both need 4.

Every one of the 7 quick-pass warnings is followed up in Section 6. All persisted in some form; the placement gap and the few-shot results improved a little with more seeds.

---

## 2. Contribution 1: SHAP-guided knowledge distillation (SHIELD)

**What this is.** A big, accurate "teacher" network (about 1.2 M parameters) trains a tiny "student" (3-4 K parameters) to copy its answers (this is
*knowledge distillation*, KD). SHIELD adds two ideas: (a) the student only looks at the **16 features the teacher found most important** (using SHAP, a method
that measures how much each input feature matters), and (b) an extra loss term (weight gamma = 0.1) that pushes the student to **use features the same way the teacher does**.

### 2.1 Accuracy on the 8 main tasks (test macro-F1 %, mean ± std over 5 seeds)

| Task | Teacher | XGBoost | Decision tree | Log. reg. | scratch_all (45 feat.) | kd_all (45) | kd_shap (16) | **SHIELD (16)** | kd_random (16) | kd_mi (16) | kd_variance (16) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIoT binary | 94.29 ± 0.49 | 95.11 | 92.59 | 89.16 | 94.64 ± 0.06 | 94.17 ± 0.12 | 93.85 ± 0.09 | **93.83 ± 0.01** | 90.98 ± 1.81 | 92.94 ± 0.06 | 93.72 ± 0.05 |
| CICIoT family (8) | 71.74 ± 0.40 | 72.93 | 59.59 | 59.17 | 67.23 ± 0.19 | 66.32 ± 0.38 | 61.78 ± 0.57 | **61.55 ± 0.31** | 58.49 ± 4.96 | 60.41 ± 0.27 | 61.01 ± 0.18 |
| CICIoT 34-class | 67.42 ± 1.97 | 68.56 | 50.01 | 55.50 | 62.96 ± 0.14 | 60.98 ± 0.35 | 55.65 ± 0.29 | **55.65 ± 0.12** | 50.83 ± 3.94 | 55.17 ± 0.23 | 50.64 ± 0.14 |
| CICIoMT binary | 97.93 ± 0.30 | 96.71 | 94.97 | 95.43 | 97.08 ± 0.50 | 97.44 ± 0.54 | 96.98 ± 0.39 | **97.11 ± 0.52** | 96.46 ± 1.00 | 95.78 ± 0.55 | 96.52 ± 0.31 |
| CICIoMT category | 80.73 ± 0.54 | 79.75 | 78.37 | 75.54 | 81.22 ± 1.72 | 80.50 ± 1.13 | 80.05 ± 1.80 | **80.23 ± 0.89** | 77.89 ± 2.22 | 79.45 ± 1.67 | 80.35 ± 1.36 |
| CICIoMT attack | 70.48 ± 0.17 | 70.39 | 61.36 | 60.11 | 66.31 ± 0.49 | 67.09 ± 0.73 | 64.59 ± 0.36 | **64.82 ± 0.48** | 60.32 ± 3.55 | 65.54 ± 1.24 | 65.59 ± 0.85 |
| Bot-IoT binary | 97.15 ± 0.69 | 98.99 | 98.12 | 86.02 | 97.30 ± 0.33 | 97.65 ± 0.48 | 97.36 ± 0.33 | **97.04 ± 0.50** | 91.03 ± 8.35 | 96.85 ± 0.46 | 96.94 ± 0.16 |
| Bot-IoT category (5) | 91.57 ± 0.56 | 95.63 | 85.72 | 65.74 | 91.73 ± 0.44 | 90.05 ± 0.57 | 87.17 ± 1.30 | **86.69 ± 1.17** | 81.21 ± 8.68 | 87.12 ± 1.57 | 87.95 ± 1.44 |

*How to read the columns.* `scratch_all` = small model trained on its own with all features; `kd_all` = distilled with all features; `kd_shap` = distilled with SHIELD's 16 SHAP
features but **no** attribution loss; `kd_random/mi/variance` = 16 features picked randomly / by mutual information / by variance (other ways to choose features).
Source: `outputs/paper/tables/detection_<dataset>_<task>.csv`.

*Side note:* on CICIoT 34-class, SHIELD and kd_shap have the same mean (55.65) by coincidence. Their per-seed scores differ
(SHIELD 55.58/55.75/55.63/55.50/55.78, kd_shap 55.27/56.07/55.62/55.54/55.74).

### 2.2 What the numbers say

- **SHIELD vs kd_shap (does the attribution loss help?):** the mean difference over all 40 seed-task pairs is **-0.07 points**, Wilcoxon p = 0.64. Per task,
  the difference is between -0.48 and +0.23 points, always smaller than the ±. **No measurable accuracy benefit.**
- **Critical-difference (CD) analysis** (Friedman test then Nemenyi; it says which methods are *statistically tied*, CD = 1.66 rank units, 40 blocks, Friedman p = 2.8e-28).
  Average ranks (lower is better): scratch_all 2.00, kd_all 2.25, **SHIELD 4.10**, kd_shap 4.15, kd_variance 4.78, kd_mi 5.50, kd_random 6.23, shield_int8 7.00.
  - Tied at the top: **scratch_all and kd_all**.
  - SHIELD is in a tied group with **kd_shap, kd_variance and kd_mi**: it is statistically better than kd_random and int8, but not better than the simpler feature selectors.
  - Figure: `outputs/paper/figures/cd_kd_ablation.png`.
- **SHAP-guided selection vs random:** SHAP features are clearly better than random ones (and much more stable; random has ± up to 8.7).
- **Feature-ranking fidelity** (does the student rely on features the way the teacher does? Spearman correlation, seed 0):
  SHIELD 0.54-0.64 vs kd_shap 0.36-0.52 on CICIoT (SHIELD better). On CICIoMT binary/category SHIELD is clearly lower (0.53 / 0.61 vs 0.68 / 0.74);
  on CICIoMT attack and both Bot-IoT tasks they are about equal (0.87 vs 0.88, 0.66 vs 0.67, 0.28 vs 0.30).
- **Simple-model check:** on Bot-IoT binary, a 10-level decision tree (98.12) beats SHIELD (97.04 ± 0.50).
- **8-bit (int8) version of SHIELD:** 2x smaller on disk (about 8 KB) but loses accuracy: drop 0.5-0.6 points on the binary tasks, 3.2-3.8 on Bot-IoT binary / CICIoT family,
  5.9-6.1 on CICIoT 34-class / CICIoMT category, **9.7 ± 4.5 on Bot-IoT category and 12.2 ± 4.7 on CICIoMT attack**. The all-feature students lose even more.

### 2.3 Speed and size (single CPU core, batch 256; from the quick pass, CPU idle)

| Task | Teacher (flows/s) | SHIELD fp32 | SHIELD int8 | SHIELD vs teacher | Size fp32 -> int8 |
|---|---:|---:|---:|---:|---|
| CICIoT binary | 48,703 | 6,564,131 | 1,462,023 | 135x | 15.3 -> 8.1 KB |
| CICIoT family | 48,411 | 6,614,996 | 1,500,587 | 137x | 16.1 -> 8.3 KB |
| CICIoT 34-class | 48,197 | 5,981,327 | 1,490,972 | 124x | 19.5 -> 9.3 KB |
| CICIoMT binary | 49,004 | 6,513,964 | 1,491,842 | 133x | 15.3 -> 8.1 KB |
| CICIoMT category | 48,653 | 6,481,020 | 1,488,372 | 133x | 15.8 -> 8.3 KB |
| CICIoMT attack | 48,280 | 6,243,890 | 1,485,780 | 129x | 17.5 -> 8.7 KB |
| Bot-IoT binary | 48,942 | 6,580,946 | 1,503,229 | 134x | 15.3 -> 8.1 KB |
| Bot-IoT category | 48,867 | 6,514,003 | 1,496,637 | 133x | 15.7 -> 8.2 KB |

The teacher model is about 4.7 MB. **int8 is about 4x slower than fp32** for a model this tiny, so present int8 as "smaller, not faster".

**Honest summary of contribution 1:** distillation into a 16-feature student gives a 130x speed-up at a modest accuracy cost on binary tasks. The SHAP feature
selection is a sensible, stable choice (much better than random, similar to MI/variance). The SHAP *attribution loss* does not measurably improve accuracy;
its only benefit is higher explanation fidelity on CICIoT.

---

## 3. Data leakage findings (why the "strict protocol" exists)

**What this is.** Some dataset columns give away the answer for reasons that have nothing to do with the network traffic itself. An example is a
counter of when the capture happened. A model can "cheat" with such a column. The strict protocol removes IAT (both CIC datasets) and `seq` (Bot-IoT).
After removing them, rows that become exact copies are dropped, and copies with different labels are removed.

### 3.1 Single-feature audit (standard protocol, seed 0)

| Audit | All features together | Leaky feature | Its rank | Its score alone | Any other suspect (>= 80% of all-features)? |
|---|---:|---|---:|---:|---|
| CICIoT 34-class | 47.5 | IAT | **1st** | **53.0** (beats all features together!) | none |
| CICIoMT attack | 57.5 | IAT | **1st** | **65.4** (beats all features together!) | none |
| Bot-IoT category | 77.9 | seq | 25th | 15.3 | none |

Source: `outputs/leakage/summary.json`, `outputs/paper/tables/leakage_audit_*.csv`.

### 3.2 How much the leak inflates results ("inflation" = standard minus strict, seed 0)

| Task | Teacher | SHIELD |
|---|---:|---:|
| CICIoT family | -1.10 | -0.42 |
| CICIoT 34-class | **+1.31** | **+2.03** |
| CICIoMT category | -0.21 | -0.42 |
| CICIoMT attack | **+1.67** | **+2.32** |
| Bot-IoT category | -0.65 | +0.58 |

Source: `outputs/paper/tables/leakage_inflation.csv`, figure `leakage_inflation.png`. The standard-protocol models were trained with seed 0 only,
so this is a single-seed comparison; differences under about 1 point are noise.

**Plain meaning:** IAT is a strong shortcut on its own, but once the model has all the other features the shortcut adds only about 1.3-2.3 points on the
fine-grained tasks. `seq` is not a measurable shortcut here. The paper should justify dropping it by what it *is* (a record counter that reflects the recording order)
and by the de-duplication effect: dropping `seq` turns 54% of Bot-IoT rows into exact duplicates (39.5 M of 73.4 M).

### 3.3 Data after the strict protocol

| Dataset | Train / validation / test rows | Duplicates removed | Conflicting-label rows removed |
|---|---|---:|---:|
| CICIoT2023 (subset, see Limitations) | 4,034,235 / 728,213 / 705,401 | 2,363,576 | 113,396 |
| CICIoMT2024 | 3,846,494 / 646,417 / 887,169 | 3,391,891 | 859,567 |
| Bot-IoT | 23,699,492 / 5,078,464 / 5,078,459 | 39,459,785 | 4,561,131 |

No main dataset is missing any class in its training split. Source: `outputs/data_report_*.json`.

---

## 4. Contribution 2: cross-dataset transfer and recovery

**What this is.** CICIoT2023 (general IoT) and CICIoMT2024 (medical IoT) use the same feature extractor. Models were trained on one using only the shared
features and 5 shared attack classes (`shared5`), then tested on the other ("zero-shot"). *Normalisation* means scaling the features using
statistics from the **source** dataset or the **target** dataset. *Few-shot* means retraining with a small share (0.1-10%) of labelled target data.

### 4.1 Zero-shot (3 seeds; in-domain -> other dataset)

| Direction, task | Teacher | XGBoost | kd_all | SHIELD |
|---|---|---|---|---|
| CICIoMT -> CICIoT, binary (in-domain about 0.97) | src 0.632 ± 0.024 / tgt 0.831 ± 0.007 | 0.832 / 0.696 | 0.738 ± 0.008 / 0.772 ± 0.056 | **0.826 ± 0.009 / 0.863 ± 0.005** |
| CICIoMT -> CICIoT, shared5 (in-domain about 0.78) | 0.307 / 0.366 | 0.454 / 0.399 | 0.431 / 0.436 | **0.476 ± 0.017** / 0.412 |
| CICIoT -> CICIoMT, binary (in-domain about 0.93) | 0.678 / 0.558 | 0.547 / 0.490 | **0.772 ± 0.029** / 0.555 | 0.552 ± 0.009 / 0.506 |
| CICIoT -> CICIoMT, shared5 (in-domain about 0.78) | 0.394 / 0.355 | 0.426 / 0.373 | 0.487 / 0.336 | 0.468 ± 0.100 / 0.361 |

(src = source normalisation, tgt = target normalisation; macro-F1 as a fraction.)

- Every model loses a lot: **11-47 points**.
- SHIELD is the best zero-shot model from CICIoMT -> CICIoT (both tasks). From CICIoT -> CICIoMT, kd_all is best and SHIELD is among the worst on binary.
- **Target normalisation helps in only 6 of 24 cases**, all in the CICIoMT -> CICIoT direction. In the other direction it always hurts.
- **Feature-importance agreement between the two datasets:** binary Spearman 0.75 (top-10 overlap 8/10); shared5 Spearman 0.88 (8/10). Both datasets
  rely on similar features (Header_Length, Rate/Srate, syn/rst counts, TCP).

### 4.2 Few-shot recovery (3 seeds): fine-tuned SHIELD vs training from scratch on the same target data

| Direction, task | 0.1% | 1% | 5% | 10% |
|---|---|---|---|---|
| CICIoMT -> CICIoT, binary | 0.873 vs 0.881 (**-0.8**) | 0.894 vs 0.891 | 0.906 vs 0.903 | 0.913 vs 0.912 |
| CICIoMT -> CICIoT, shared5 | **0.666 vs 0.491 (+17.4)** | **0.686 vs 0.570 (+11.6)** | 0.724 vs 0.712 | 0.735 vs 0.733 |
| CICIoT -> CICIoMT, binary | 0.956 vs 0.962 (**-0.6**) | 0.967 vs 0.958 | 0.971 vs 0.962 | 0.969 vs 0.961 |
| CICIoT -> CICIoMT, shared5 | 0.684 vs 0.671 | **0.713 vs 0.665 (+4.8)** | 0.751 vs 0.744 | 0.760 vs 0.747 |

- Fine-tuning helps most for the **5-class task with very little data** (0.1-1%). That is the useful "recovery" result.
- Fine-tuned is **not** better than scratch in 6 of 32 cells (SHIELD and kd_all combined), mostly at 0.1% on binary, where scratch training is already good.
- Source: `outputs/cross/fewshot_summary.csv`, figures `fewshot_*.png`.

### 4.3 Never-seen attack families (binary detector, detection rate, source normalisation)

- CICIoMT -> CICIoT: **Mirai** is caught by almost everything (SHIELD 0.996). **BruteForce** (SHIELD 0.07) and **Web** (SHIELD 0.08) are mostly missed by every
  neural model. XGBoost with target normalisation is the exception (0.76 / 0.85).
- CICIoT -> CICIoMT: **MQTT** attacks are caught by every model (0.98-1.00).

**Honest summary of contribution 2:** cross-dataset generalisation is poor without adaptation. SHIELD is the best zero-shot model in one direction only.
Small amounts of target data recover a lot on the multi-class task. Rare application-layer attacks (BruteForce, Web) that were never seen are not detected.

---

## 5. Contribution 3: hybrid EHO-ACO controller placement, plus coupling

**What this is.** In a software-defined network, *controllers* are the computers that run the detector for many switches. *Placement* chooses where to put
k controllers so that switches reach them quickly (low delay) without overloading them. EHO-ACO is the proposed hybrid search method, compared with 11 others.
The *optimality gap* is how far a method's answer is from the true best answer (found by brute force on small networks).

### 5.1 Ranking (30 runs per setting, 56 problem instances)

- **Friedman test p = 3.5e-76** (the methods really differ). CD = 2.23 rank units.
- Average ranks: **SA 1.77**, ACO 3.32, **hybrid EHO-ACO 4.00**, EHO 4.05, GA 4.51, random 4.74, PSO 5.82, ILP 8.80, k-median 9.35, k-means 9.81, PageRank 10.36, k-center 11.48.
- **Statistically tied with the best:** SA, ACO and **the hybrid**. The hybrid is also tied with EHO, GA and random search. Figure: `outputs/paper/figures/cd_placement.png`.
- **Large networks** (mean rank across settings): Cogentco: SA 1.64, ACO 3.21, **hybrid 3.64** (tied 3rd with GA); Kdl: SA 1.21, ACO 2.21, **hybrid 2.93 (3rd)**;
  syn500: SA 1.43, ACO 2.00, **hybrid 3.00 (3rd)**.

### 5.2 Optimality gap on small networks (vs brute force)

| Method | Mean gap | Median | Worst |
|---|---:|---:|---:|
| SA | **0.28%** | 0.00% | 1.9% |
| random search | 0.64% | 0.01% | 3.4% |
| EHO | 0.62% | 0.05% | 3.6% |
| ACO | 1.11% | 0.11% | 9.4% |
| **hybrid EHO-ACO** | **1.46%** | 0.14% | **15.0%** |
| GA | 1.71% | 0.07% | 18.5% |
| PSO | 2.60% | 0.55% | 24.3% |
| ILP / k-median / k-means / PageRank / k-center | 42-109% | | |

The hybrid finds the exact optimum in only **32%** of small instances. Runtime per run: hybrid 0.47 s, SA 0.47 s, EHO 0.41 s, ACO 1.99 s, PSO 2.42 s, ILP 9.35 s.
Sources: `outputs/placement/summary.csv`, `outputs/paper/tables/placement_gap.csv`, `placement_wilcoxon.csv`.

### 5.3 Coupling: how many controllers are needed when they run the detector (Geant2012 network)

| Network load (rho) | SHIELD | SHIELD int8 | Teacher |
|---|---:|---:|---:|
| 0.5 | 4 | 4 | 4 |
| 0.7 | 4 | 4 | **6** |
| 0.9 | 4 | 4 | **7** |

Controller capacity: SHIELD about 12.0 M flows/s, int8 about 3.0 M, teacher about 96 K. Table `coupled_min_controllers.csv`, figure `coupled_min_controllers.png`.

**Honest summary of contribution 3:** the metaheuristics all beat the classic heuristics by a wide margin. The hybrid is statistically tied with the best,
but **SA is better on every measure**, and even random search has a smaller mean gap than the hybrid. The paper cannot claim that the hybrid is the best method;
it can claim it is competitive. Coupling shows a clear benefit of the fast detector at medium and high load (2-3 fewer controllers) and none at low load.

---

## 6. Quick-pass warnings: did they persist with more seeds?

| # | Quick-pass warning (seed 0) | Full run (5 / 3 / 30 seeds) | Status |
|---|---|---|---|
| 1 | Bot-IoT `seq` is not a strong single-feature leak | Audit is single-seed by design; unchanged | **Persists** |
| 2 | Decision tree >= SHIELD on Bot-IoT; kd_variance > SHIELD on 3 tasks | dtree 98.12 vs SHIELD 97.04 ± 0.50 (binary) and 85.72 vs 86.69 ± 1.17 (category). kd_variance > SHIELD on CICIoMT category/attack and Bot-IoT category, all within ±. | **Persists for Bot-IoT binary; category now within noise** |
| 3 | Large int8 accuracy drops; int8 slower than fp32 | SHIELD int8 drops 0.5-12.2 points (worst CICIoMT attack 12.2 ± 4.7); 4x slower | **Persists** |
| 4 | Hybrid not near-optimal; SA better | Hybrid mean gap 1.46% (was 2.65%), worst 15% (was 34%); exact optimum in 32% (was 45%); SA still best | **Persists (gap smaller)** |
| 5 | Teacher = SHIELD controllers at rho 0.5 | Unchanged (4 vs 4) | **Persists** |
| 6 | Fine-tuned not always > scratch; target normalisation rarely helps | Not better in 6 of 32 (was 8); target norm better in 6 of 24 (was 3), all in one direction | **Persists, slightly better** |
| 7 | CICIoT2023 input is a subset | Unchanged | **Persists** |
| new | (Quick pass: shield > kd_shap on 6 of 8, single seed) | With 5 seeds: no difference (p = 0.64) | **Became a key finding** |

---

## 7. Runtime, decisions and fixes during the run

**Runtime**

| Stage | Wall-clock |
|---|---|
| Smoke test (preflight) | 5 min |
| Quick pass (seed 0, everything) | 6 h 19 min (02:29-08:59) |
| Full run, GPU chain (teachers 86 min, SHAP 166 min, distillation 266 min, k-sweep 102 min, cross 10 min) | 10 h 30 min (09:03-19:33) |
| Full run, placement (in parallel, 12 workers) | 15 min |
| Coupling + figures | 1 min |
| **Total** | **about 17 h** |

**Decisions and fixes** (all logged in `outputs/RUN_STATE.md`; none changed the scientific protocol):
- **Storage moved to D:** at your instruction: `data/` and `outputs/` now live in `D:\shield_run\` (C: was too small). D: is a hard disk.
- **Topology Zoo:** topology-zoo.org was unreachable. The 4 topology files come from the official Zoo data store and are **byte-for-byte identical to the Internet Archive copy**.
  After cleaning: Abilene 11, Geant2012 37, Cogentco 180, Kdl 709 nodes.
- **Slow streaming of the 49 M-row standard Bot-IoT dataset:** fixed without code changes by reading the file once into Windows' RAM cache.
- **Memory pressure during Bot-IoT distillation** (heavy paging, epochs 3-4x slower): no crash, so the number of parallel jobs was left at 3, as the rules require.
- **Gamma tuning rule:** not triggered (SHIELD's validation score was below kd_shap's on only 1 of 8 tasks). gamma stayed 0.1.
- **Config housekeeping:** quick-pass seed settings were restored to the committed values before the full run (checked against git). PowerShell's
  byte-order-mark problem in YAML files was avoided. **No code changes and no commits** were needed, so nothing was pushed.

---

## 8. Honest limitations

1. **CICIoT2023 is a subset.** This machine's zip has 7.85 M rows before de-duplication; the full release has about 46 M. The paper must say which subset was used.
2. **One XGBoost / decision-tree / logistic-regression seed.** Their numbers have no error bars. Logistic regression did not fully converge on 7 fits (300-iteration cap).
3. **Standard-protocol (leaky) models used one seed**, so the inflation numbers are single-seed comparisons.
4. **SHAP fidelity is measured on seed 0 only** (by design, to save time).
5. **The CICIoMT teachers hit the 40-epoch cap** with their best epoch near the end. They might improve slightly with more epochs.
6. **Coupling uses one topology (Geant2012) and three load levels.**
7. **Placement uses the same evaluation budget for every method** (20,000). Conclusions about the hybrid may change with other budgets or settings; that was not tested.
8. **int8 quantisation** is a straightforward post-training version. Its large drops may be fixable (for example with per-channel or input-aware calibration). This was not explored.
9. **Latency** was measured once on this desktop CPU (Intel 20-thread). Absolute numbers depend on hardware; the ratios are the meaningful part.

---

## 9. Which tables and figures to use in the paper

All files are in `D:\shield_run\outputs\paper\` (`tables/` has `.csv` and `.tex` pairs; `figures/` has `.png` and `.pdf` pairs).

| Paper section | Use | Files |
|---|---|---|
| Data and leakage | Leakage audit and inflation | `tables/leakage_audit_*.tex`, `tables/leakage_inflation.tex`, `figures/leakage_inflation.png` |
| KD main results | One table per task (mean ± std) | `tables/detection_ciciot_binary.tex` ... `detection_botiot_category.tex` (8 main tasks) |
| KD statistics | Critical-difference diagram | `figures/cd_kd_ablation.png`, `tables/kd_ablation_ranks.json` |
| Feature budget | F1 vs number of features k | `figures/f1_vs_k_*.png`, `tables/k_sweep_*.tex` |
| Accuracy vs speed | Pareto plots | `figures/pareto_*.png` |
| Error analysis | Confusion matrices | `figures/confusion_shield_*.png`, `figures/confusion_teacher_*.png` |
| Cross-dataset | Zero-shot, few-shot, unseen attacks | `tables/cross_zero_shot.tex`, `tables/cross_fewshot.tex`, `tables/cross_unseen_attacks.tex`, `figures/fewshot_*.png` |
| Placement | Gap, objective, statistics, scalability | `tables/placement_gap.tex`, `tables/placement_objective.tex`, `tables/placement_wilcoxon.tex`, `figures/cd_placement.png`, `figures/convergence_*.png`, `figures/objective_vs_k_*.png`, `figures/placement_scalability.png` |
| Resilience | Controller / link failures | `tables/placement_resilience.tex`, `figures/resilience_*.png` |
| Coupling | Controllers needed vs load | `tables/coupled_min_controllers.tex`, `figures/coupled_min_controllers.png` |
| Robustness check | Standard-protocol tables | `tables/detection_*_std_*.tex` (supplementary only) |

Supporting files: `outputs/QUICK_PASS_REPORT.md` (seed-0 checks), `outputs/final_analysis/` (the script and raw numbers behind this report),
`outputs/RUN_STATE.md` (full log of everything done).

---

## 10. Decisions left for you

1. **How to frame SHIELD's attribution loss**, since it does not improve accuracy. Options: present it as an explainability-fidelity method (it helps on CICIoT), or drop the claim.
2. **How to frame the hybrid EHO-ACO**, since SA beats it. Options: present it as "competitive / tied with the best", or revisit its settings in a separate study.
3. **Whether to include int8**, or present it only as "2x smaller, less accurate, slower".
4. **Whether to re-run on the full CICIoT2023** (about 46 M rows) if the paper needs the complete dataset.
5. **How to justify removing `seq`**, given that the audit does not flag it.
