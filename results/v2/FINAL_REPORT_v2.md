# SHIELD-IoT v2: final results report (with comparison to v1)

*Written for you: someone new to machine learning and networking. Each section says what the experiment does in plain words, then gives the v2 numbers next to v1's.*

**Run:** 4 October 2026, 00:46 to about 22:33 (about 22 h wall-clock, including one planned restart), desktop RTX 3060 12 GB, 20 CPU threads, 32 GB RAM.
**Code:** v2 = commit e1f85e4. v1 = e9e82e7 (see `D:\shield_run\outputs\FINAL_REPORT.md`). No code was changed by me.
**Seeds:** teachers and students 5 seeds, cross-dataset 3 seeds, placement 30 runs per setting, the same as v1.
Unless marked otherwise, numbers are **test macro-F1 in %, mean ± standard deviation over seeds**.

> *Macro-F1* (0-100) treats every class as equally important. *±* is the spread over repeated runs; differences smaller than the ± are noise.
> In v2, every tuning choice (objective, feature count k, student size, placement settings) was made **on validation data only**. Test data was used only for the numbers in this report.

---

## 1. The short version

| | v1 | v2 |
|---|---|---|
| Student-teacher gap on the hardest tasks (CICIoT family / 34-class) | 10.2 / 11.8 points | **2.4 / 3.7 points** |
| Does SHIELD's SHAP loss beat the same model without it (`kd_shap`)? | No (p = 0.64) | **Still no** (p = 0.10, SHIELD slightly *worse*: -0.15 points) |
| Does distillation beat training the small model alone? | No (scratch_all ranked best) | **Now tied**: `kd_all` ranks 1st, scratch_all 2nd, not significantly different |
| Placement: is the hybrid the best? | No: SA best, hybrid 3rd | **Yes on rank** (1st, tied group with SA/PSO/ACO/GA); mean gap 0.46% (was 1.46%) |
| What makes the hybrid good? | — | **The local search, not the ants**: removing ants changes nothing; removing local search raises the gap to 2.1% |
| Coupling: fewer controllers with SHIELD? | 1 topology, only at load >= 0.7 | 3 topologies: **1-3 fewer controllers at high load**, equal at low load on 2 of 3 |
| int8 (8-bit) models | up to 12 points worse | **worse still for the bigger v2 students** (up to 22 points) |

**Bottom line:** v2's tuning closed most of the accuracy gap to the teacher, by choosing bigger students and more features for the hard tasks. They are still about 60x faster than the teacher. The memetic hybrid is now the top-ranked placement method. The SHAP attribution loss (SHIELD's distinctive idea) still gives no accuracy benefit.

---

## A. Teacher scores (v2 vs v1)

v2 raised the teacher's epoch cap from 40 to 80 (early stopping on validation unchanged).

| Task | v1 teacher | v2 teacher | Change |
|---|---:|---:|---:|
| CICIoT binary | 94.29 ± 0.49 | 94.15 ± 0.56 | -0.14 |
| CICIoT family | 71.74 ± 0.40 | 71.15 ± 0.74 | -0.59 |
| CICIoT 34-class | 67.42 ± 1.97 | 68.09 ± 1.19 | +0.67 |
| CICIoMT binary | 97.93 ± 0.30 | 97.71 ± 0.13 | -0.22 |
| CICIoMT category | 80.73 ± 0.54 | 80.65 ± 0.56 | -0.08 |
| CICIoMT attack | 70.48 ± 0.17 | 69.87 ± 0.78 | -0.61 |
| Bot-IoT binary | 97.15 ± 0.69 | 96.90 ± 0.66 | -0.25 |
| Bot-IoT category | 91.57 ± 0.56 | 92.26 ± 1.15 | +0.69 |

**Plain meaning:** the teachers are essentially unchanged; every difference is within about one ±. The higher cap was used by only a few runs (2 of 40 main-task teachers stopped at the cap of 80 epochs; most stop at 10-60).
XGBoost (95.11 / 72.93 / 68.56 / 96.71 / 79.75 / 70.39 / 98.99 / 95.63) is identical to v1 (same seed and settings).

---

## B. The tuning choice, and does distillation now beat training alone?

**What was tuned (on validation, seed 0):** 13 objective settings were compared, averaged over the 8 main tasks (`tables/kd_tuning_objective.csv`):

| Rank | Setting | Mean validation macro-F1 |
|---:|---|---:|
| **1** | **Plain KD, temperature T = 1, alpha = 0.7** | **84.45** |
| 2 | Plain KD, T = 1, alpha = 0.3 | 84.23 |
| 3 | Plain KD, T = 2, alpha = 0.7 | 84.05 |
| 4 | Plain KD via a teacher **assistant** (TAKD) | 83.95 |
| 6 | Best decoupled KD (DKD, T = 1, beta = 2) | 83.62 |
| 13 | DKD, T = 4, beta = 8 | 82.02 |

- **Objective chosen:** plain KD with a low temperature (T = 1). The newer DKD objective did **not** win; its best setting is 0.8 points lower.
- **Teacher vs assistant:** the original **teacher** was chosen. An intermediate "teacher assistant" model was 0.5 points worse.

**Does KD now beat training the small model on its own (`scratch_all`)?** Both use all features, so this is a fair test of distillation itself.

| Task | v1 kd_all - scratch_all | v2 kd_all - scratch_all | v2 values |
|---|---:|---:|---|
| CICIoT family | -0.91 | **+0.28** | 69.35 ± 0.30 vs 69.08 ± 0.09 |
| CICIoT 34-class | -1.99 | -0.30 | 65.18 ± 0.33 vs 65.48 ± 0.15 |
| Bot-IoT category | -1.68 | **+0.60** | 92.74 ± 1.54 vs 92.14 ± 1.23 |

**Plain meaning:** in v1, distillation *hurt* on these three hard tasks. In v2 it no longer hurts: it is slightly ahead on CICIoT family and Bot-IoT category, and slightly behind on 34-class. **All three differences are within the ±**, so the honest statement is *KD now matches training alone on the hard tasks, and is ahead on 7 of 8 tasks overall (all but 34-class, mostly by less than the ±)*.
Across all 8 tasks, kd_all is the top-ranked method (Section C).

---

## C. Ranking, per-task k and size, and the student-teacher gap

### C.1 Per-task choices (on validation: the cheapest setting within 0.5 points of the best)

All tasks use the **class-balanced** SHAP ranking.

| Task | k (features) | Student layers | Parameters | v1 (k / params) |
|---|---:|---|---:|---|
| CICIoT binary | 16 | 64-32 | 3,234 | 16 / 3,234 |
| CICIoT family | **32** | **128-64** | 13,000 | 16 / 3,432 |
| CICIoT 34-class | **32** | **128-64** | 14,690 | 16 / 4,290 |
| CICIoMT binary | 12 | 64-32 | 2,978 | 16 / 3,234 |
| CICIoMT category | 16 | **128-64** | 10,822 | 16 / 3,366 |
| CICIoMT attack | **32** | **128-64** | 13,715 | 16 / 3,795 |
| Bot-IoT binary | 12 | 64-32 | 2,978 | 16 / 3,234 |
| Bot-IoT category | 12 | **128-64** | 10,245 | 16 / 3,333 |

The tuning made the easy (binary) students *smaller*, and the hard multi-class students 3-4x bigger with up to 32 features.

### C.2 Accuracy (test macro-F1 %, mean ± std over 5 seeds)

| Task | Teacher | scratch_all | kd_all | kd_shap | **SHIELD** | kd_random | kd_mi | kd_variance |
|---|---|---|---|---|---|---|---|---|
| CICIoT binary | 94.15 ± 0.56 | 94.64 ± 0.06 | 94.72 ± 0.04 | 94.47 ± 0.04 | **94.28 ± 0.02** | 91.48 ± 1.84 | 93.39 ± 0.04 | 94.11 ± 0.04 |
| CICIoT family | 71.15 ± 0.74 | 69.08 ± 0.09 | 69.35 ± 0.30 | 68.90 ± 0.05 | **68.78 ± 0.33** | 67.45 ± 1.84 | 67.44 ± 0.46 | 67.97 ± 1.30 |
| CICIoT 34-class | 68.09 ± 1.19 | 65.48 ± 0.15 | 65.18 ± 0.33 | 64.22 ± 0.37 | **64.35 ± 0.30** | 63.23 ± 2.03 | 64.60 ± 0.23 | 65.18 ± 0.35 |
| CICIoMT binary | 97.71 ± 0.13 | 97.08 ± 0.50 | 97.90 ± 0.48 | 96.81 ± 0.27 | **96.61 ± 0.34** | 96.65 ± 1.42 | 95.70 ± 0.39 | 96.50 ± 0.40 |
| CICIoMT category | 80.65 ± 0.56 | 82.69 ± 1.04 | 83.20 ± 1.56 | 82.00 ± 0.51 | **82.83 ± 1.67** | 80.12 ± 1.64 | 81.30 ± 1.52 | 81.81 ± 1.58 |
| CICIoMT attack | 69.87 ± 0.78 | 68.56 ± 1.42 | 70.44 ± 0.47 | 69.88 ± 0.28 | **69.24 ± 1.03** | 68.83 ± 1.15 | 69.83 ± 0.51 | 70.06 ± 0.50 |
| Bot-IoT binary | 96.90 ± 0.66 | 97.22 ± 0.34 | 97.24 ± 0.29 | 96.56 ± 0.49 | **96.07 ± 0.32** | 89.85 ± 9.33 | 95.47 ± 0.88 | 79.01 ± 2.48 |
| Bot-IoT category | 92.26 ± 1.15 | 92.14 ± 1.23 | 92.74 ± 1.54 | 91.96 ± 1.25 | **91.46 ± 0.92** | 85.30 ± 7.23 | 92.23 ± 1.10 | 76.18 ± 1.50 |

Source: `outputs_v2/paper/tables/detection_<dataset>_<task>.csv`. The k/size above applies to all the 16-feature-style variants (SHIELD, kd_shap, kd_random, kd_mi, kd_variance).

### C.3 Student-teacher gap (teacher minus SHIELD, points)

| Task | v1 gap | v2 gap |
|---|---:|---:|
| CICIoT binary | 0.46 | -0.13 (student ahead) |
| CICIoT family | **10.19** | **2.36** |
| CICIoT 34-class | **11.77** | **3.74** |
| CICIoMT binary | 0.82 | 1.10 |
| CICIoMT category | 0.50 | -2.17 (student ahead) |
| CICIoMT attack | 5.66 | 0.63 |
| Bot-IoT binary | 0.10 | 0.82 |
| Bot-IoT category | 4.88 | 0.80 |

**Prediction agreement** (how often SHIELD gives the same answer as the teacher on test data): 0.99-1.00 on the binary tasks, 0.95-0.97 on CICIoT family/34-class, CICIoMT category and Bot-IoT category, and 0.87 on CICIoMT attack.

### C.4 Statistical ranking (critical-difference diagram)

Friedman p = 3.4e-22 (40 blocks, CD = 1.66 rank units). Average ranks (lower is better):
**kd_all 1.95**, scratch_all 2.85, kd_shap 3.96, **SHIELD 4.18**, kd_variance 5.08, kd_mi 5.36, kd_random 5.95, shield_int8 6.68.

- Tied at the top: **kd_all and scratch_all**.
- SHIELD is now in a tied group **with scratch_all and kd_shap** (in v1 it was only tied with kd_shap/variance/MI), so the gap to the best has narrowed.
- Figure: `outputs_v2/paper/figures/cd_kd_ablation.png`. v1 order for comparison: scratch_all 2.00, kd_all 2.25, SHIELD 4.10, kd_shap 4.15.

### C.5 Does the SHAP attribution loss help? (SHIELD vs kd_shap, same features and size)

Paired Wilcoxon over 40 seed-task pairs: **p = 0.10, mean difference -0.15 points, SHIELD better in 15 of 40.** In v1 it was p = 0.64, -0.07, 20 of 40.
**Still no benefit**; if anything the attribution loss is slightly worse in v2 (not significant).

### C.6 Speed and size (single CPU core, batch 256)

| Task | Teacher (flows/s) | SHIELD fp32 | vs teacher | SHIELD size (fp32 -> int8) |
|---|---:|---:|---:|---|
| CICIoT binary | 48,735 | 6,649,360 | 136x | 15.3 -> 8.1 KB |
| CICIoT family | 48,792 | 3,091,789 | 63x | 53.5 -> 17.9 KB |
| CICIoT 34-class | 48,211 | 2,853,954 | 59x | 60.1 -> 19.7 KB |
| CICIoMT binary | 49,323 | 6,790,462 | 138x | 14.3 -> 7.9 KB |
| CICIoMT category | 48,966 | 3,427,042 | 70x | 45.0 -> 15.8 KB |
| CICIoMT attack | 48,570 | 3,011,766 | 62x | 56.3 -> 18.7 KB |
| Bot-IoT binary | 48,711 | 6,772,476 | 139x | 14.3 -> 7.9 KB |
| Bot-IoT category | 49,008 | 3,585,441 | 73x | 42.7 -> 15.3 KB |

**The trade-off v2 made:** the bigger students on the hard tasks are about half as fast as v1's (2.9-3.6 M vs 6.0-6.6 M flows/s). They are still **59-73x faster than the teacher** and under 61 KB.

---

## D. Low-data distillation (0.1%, 1%, 10% of the training data)

**What this is:** the same comparison (SHIELD vs kd_shap) when the student only sees a small fraction of the training data, where a better training signal should matter most.
Paired one-sided Wilcoxon over 40 seed-task pairs (`tables/lowdata_wilcoxon.csv`):

| Fraction | Macro-F1: mean diff (points) | SHIELD wins | p | Agreement with teacher: mean diff | wins | p |
|---|---:|---:|---:|---:|---:|---:|
| 0.1% | +0.12 | 21 / 40 | 0.34 | +0.12 | 22 / 40 | 0.16 |
| 1% | -0.03 | 18 / 40 | 0.56 | +0.09 | 17 / 40 | 0.41 |
| 10% | +0.01 | 20 / 40 | 0.46 | -0.15 | 15 / 40 | 0.77 |

**Plain meaning:** even with very little data, the attribution loss makes **no significant difference** to accuracy or to how often the student agrees with the teacher.
The one task where SHIELD is consistently ahead is CICIoMT category (for example 78.19 vs 76.62 at 1%, 76.04 vs 75.08 at 0.1%). Bot-IoT binary goes the other way at 1% and 10% (92.72 vs 93.50, 95.27 vs 96.58).
Source: `tables/lowdata_summary.csv`. (v1 had no low-data experiment.)

---

## E. Cross-dataset adaptation: CORAL and DANN vs the earlier normalisations

**What this is:** train on one CIC dataset, test on the other without labels from the new dataset. v1 tried scaling features with source or target statistics. v2 adds two standard *unsupervised adaptation* methods:
**CORAL** (re-aligns the feature correlations of the new dataset to the old one) and **DANN** (trains the network so it cannot tell the two datasets apart; only run for kd_all and SHIELD).

Zero-shot macro-F1 (fraction, mean of 3 seeds):

| Direction, task | Model | Source norm | Target norm | CORAL | DANN |
|---|---|---:|---:|---:|---:|
| CICIoMT -> CICIoT binary | SHIELD | 0.827 | **0.857** | 0.833 | 0.810 |
| | kd_all | 0.775 | 0.711 | **0.814** | 0.803 |
| CICIoMT -> CICIoT shared5 | SHIELD | 0.506 | 0.465 | 0.453 | **0.527** |
| | kd_all | 0.366 | 0.396 | 0.444 | **0.462** |
| CICIoT -> CICIoMT binary | SHIELD | 0.609 | 0.571 | 0.779 | **0.859** |
| | kd_all | 0.643 | 0.510 | **0.816** | 0.701 |
| CICIoT -> CICIoMT shared5 | SHIELD | 0.391 | 0.338 | 0.439 | **0.599** |
| | kd_all | 0.430 | 0.308 | 0.394 | **0.588** |

- **CORAL** beats source normalisation in 16 of 24 model/direction/task cases, and target normalisation in 16 of 24. It is the single best choice in 9 of 24 cases. It does *not* help XGBoost (always worse).
- **DANN** beats both earlier normalisations in **7 of 8** cases. For SHIELD it lifts CICIoT -> CICIoMT binary from 0.61 to **0.86** and the 5-class task from 0.39 to **0.60**: the biggest cross-dataset improvement in either version.
- In v1 the best fix (target normalisation) helped in only 6 of 24 cases and always in one direction. **v2's adaptation methods are a clear improvement.**
- **Few-shot fine-tuning** (labelled target data) is not better than scratch in only 4 of 48 cells (v1: 6 of 32).
- **Never-seen attack families** are still missed by SHIELD: BruteForce 0.035 and Web 0.05-0.09 (XGBoost with target norm: 0.76 / 0.85). Mirai (0.96-1.00) and MQTT (0.99-1.00) are caught.
- **Feature-importance agreement** between the two datasets: Spearman 0.84 (binary) and 0.83 (shared5), top-10 overlap 7/10 (v1: 0.75 / 0.88, 8/10).

---

## F. Placement: ablation, tuned settings, gap and ranks (hybrid vs SA)

**What changed in v2:** the hybrid became *memetic*: it adds a simulated-annealing-style local search step. Every metaheuristic was tuned with the **same effort on separate tuning graphs** (not on the test networks), so the comparison is fair.

### F.1 Tuned settings (`tables/placement_tuning.csv`; lower score is better, 1.0 = best known)

| Method | Tuned score | Default score | Tuned settings |
|---|---:|---:|---|
| **hybrid EHO-ACO** | **1.0078** | 1.0332 | 3 clans x 8, alpha 0.5, beta 0.3, ACO b 4, evaporation 0.2, local search every 2 iterations x 500 evaluations, SA start temperature 5% |
| PSO | 1.0277 | 1.0964 | swarm 60, w 0.2, c1 0.5, c2 0.1 |
| GA | 1.0308 | 1.1080 | population 40, crossover 0.95, mutation 0.4, tournament 2 |
| SA | 1.0319 | 1.0486 | cooling 0.9995, 100 start samples |
| ACO | 1.0489 | 1.0840 | 20 ants, a 0.5, b 2, evaporation 0.2 |
| EHO | 1.0810 | 1.1018 | 3 clans x 5, alpha 0.7, beta 0.1 |

The ablation variants (no ants / no local search / v1 swap search) reuse the hybrid's tuned settings.

### F.2 Ranking and statistics (30 runs, 56 instances)

Friedman p = 1.4e-96, CD = 2.87. Average ranks: **hybrid 3.44**, hybrid without ants 3.63, SA 3.67, PSO 4.02, ACO 4.94, GA 5.23, EHO 6.52, random 7.87, hybrid without local search 7.94, hybrid with v1 swap search 8.02, ILP 11.75, k-median 12.35, k-means 12.76, PageRank 13.38, k-center 14.48.

- **The hybrid is now ranked 1st** (v1: 3rd, behind SA and ACO).
- **Statistically tied group at the top:** hybrid, hybrid without ants, SA, PSO, ACO and GA. Figure `cd_placement.png`.
- **Hybrid vs SA, per instance** (Wilcoxon over 30 runs, Holm-corrected): the hybrid is significantly better on **8 of 56** instances and significantly worse on **0**. Over all runs: 376 wins, 1,071 ties, 233 losses.
- **Large networks** (mean rank): Kdl: **hybrid 1st** (1.64; SA 2.57). syn500: **hybrid 1st** (1.79; SA 3.14). Cogentco: PSO 1st (2.64), SA 3.21, hybrid 4th (3.86).

### F.3 Optimality gap on small networks (vs brute force)

| Method | Mean gap (v2) | Median | Worst | Exact optimum found | v1 mean gap |
|---|---:|---:|---:|---:|---:|
| PSO | **0.11%** | 0.00% | 1.0% | — | 2.60% |
| SA | 0.13% | 0.00% | 1.6% | 68% | 0.28% |
| hybrid without ants | 0.27% | 0.00% | 2.9% | 82% | — |
| ACO | 0.34% | 0.00% | 2.2% | — | 1.11% |
| EHO | 0.40% | 0.02% | 3.1% | — | 0.62% |
| **hybrid EHO-ACO** | **0.46%** | **0.00%** | 6.6% | **82%** | **1.46%** |
| random search | 0.64% | 0.01% | 3.4% | — | 0.64% |
| GA | 0.67% | 0.00% | 6.6% | — | 1.71% |
| hybrid with v1 swap search | 1.84% | 0.25% | 16.6% | — | — |
| hybrid without local search | 2.10% | 0.42% | 22.4% | — | — |

### F.4 Ablation (what each part of the hybrid contributes; `tables/placement_ablation.csv`)

| Variant | Mean objective | Mean gap | vs full hybrid |
|---|---:|---:|---|
| Full memetic hybrid | 3.740 | 0.46% | — |
| Without ants | 3.762 | 0.27% | tied: significantly different on only 4 of 56 instances |
| Without local search | 3.857 | 2.10% | clearly worse (hybrid significantly better on 42 of 56) |
| With v1 swap search | 3.875 | 1.84% | clearly worse (39 of 56) |

**Plain meaning:** the hybrid is now the top-ranked method and finds the true optimum on 82% of small networks (v1: 32%). **The improvement comes from the new SA-style local search.** The ants (ACO part) add nothing measurable: the version without them performs the same.
SA and PSO still have a slightly lower *mean* gap on small networks, but the hybrid wins on the large networks Kdl and syn500. Runtime per run: hybrid 0.85 s, SA 0.73 s, PSO 2.46 s.

---

## G. Coupling on all three topologies (with their latency target, SLA)

**What this is:** how many controllers each network needs when the controllers run the detector. The SLA is now relative to each topology: 1.25x the 95th-percentile delay.
Controller capacity (CICIoT 34-class model): SHIELD 5.7 M flows/s, SHIELD int8 2.4 M, teacher 96 K.

| Topology (SLA) | Load | SHIELD | SHIELD int8 | Teacher | Saving |
|---|---:|---:|---:|---:|---:|
| Geant2012 (20.8 ms) | 0.5 | 3 | 3 | 4 | 1 |
| | 0.7 | 3 | 3 | 6 | **3** |
| | 0.9 | 3 | 3 | 7 | **4** |
| Cogentco (215.8 ms) | 0.5 | 6 | 6 | 6 | 0 |
| | 0.7 | 6 | 6 | 6 | 0 |
| | 0.9 | 6 | 6 | 8 | **2** |
| syn200 (70.9 ms) | 0.5 | 7 | 7 | 7 | 0 |
| | 0.7 | 7 | 7 | 7 | 0 |
| | 0.9 | 7 | 7 | 8 | **1** |

**Plain meaning:** on Geant2012, SHIELD saves controllers at every load, which is better than v1, where it saved none at load 0.5. On the large networks, the number of controllers is set by the delay target (distance), not by processing speed, so SHIELD only helps at the highest load (2 and 1 fewer).
Files: `tables/coupled_min_controllers.csv`, `figures/coupled_min_controllers.png`.

---

## H. int8 (8-bit) models

SHIELD int8 accuracy drop (fp32 minus int8, points, mean over 5 seeds):

| Task | v1 drop | v2 drop |
|---|---:|---:|
| CICIoT binary | 0.63 | 0.31 |
| CICIoT family | 3.76 | **18.69** |
| CICIoT 34-class | 5.86 | 6.36 |
| CICIoMT binary | 0.45 | -0.01 |
| CICIoMT category | 6.12 | 2.14 |
| CICIoMT attack | 12.22 | 10.29 |
| Bot-IoT binary | 3.17 | 0.75 |
| Bot-IoT category | 9.67 | **21.85** |

int8 throughput: 1.2-1.5 M flows/s, about 2-5x **slower** than fp32 and about 3x smaller on disk.
**Plain meaning:** the small binary students quantise fine (drop under 1 point), but the bigger multi-class students on CICIoT family and Bot-IoT category lose about 20 points. int8 remains "smaller, not faster", and it is unreliable for the larger models. int8 is also ranked last in the critical-difference diagram.

---

## 9. Honest limitations

1. **CICIoT2023 is a subset** (7.85 M of about 46 M rows), as in v1.
2. **The v2 tuning was done on seed 0 only**, then fixed for all seeds. It used validation data only (no test data), but a different tuning seed could pick a different k or size.
3. **One test failed before the run** (56 of 57 passed): `test_hybrid_finds_brute_force_optimum` expects the hybrid to hit the optimum on at least 29 of 30 seeds on a small graph; on this machine it hits 28. The two misses are 0.009% above the optimum. It is a borderline threshold, not a bug, and was left unchanged.
4. **The bigger students are slower**: 3x the parameters of v1 for the hard tasks, and half the speed.
5. **kd_variance collapses on Bot-IoT in v2** (79.0 and 76.2 vs about 97 / 88 in v1), because at the tuned k = 12 the variance-based selection keeps uninformative features. This is a property of that baseline, not of SHIELD, but it inflates SHIELD's rank against it.
6. **The ablation variants reuse the hybrid's tuned settings** rather than being tuned separately.
7. **DANN was run for kd_all and SHIELD only**, not for the teacher or XGBoost.
8. **Single XGBoost / decision-tree / logistic-regression seed, single-seed leakage and standard-protocol models, seed-0-only SHAP fidelity** (as in v1). The leakage findings are unchanged from v1.
9. **One planned restart during the run:** kd.yaml `jobs` was lowered from 3 to 2 because of heavy RAM paging (allowed fix). This only changes how many runs train at once, not results. Finished runs were kept.

---

## 10. Runtime and what I did during the run

| Stage | Time |
|---|---|
| Smoke test (v2 code, 18 steps) | 5.8 min |
| Teachers (57) | 2.2 h |
| XGBoost + baselines | 50 min |
| SHAP (class-balanced, 57 teachers) | 4.4 h |
| KD tuning (objective, teacher/assistant, k/size: 272 runs) | 4.9 h |
| Distillation (333 runs) | about 4 h (restart at run 224) |
| k-sweep (160) | 3.0 h |
| Low-data (240) | about 30 min |
| Latency, placement tuning, placement (18,704 runs), coupling, cross-dataset (with CORAL/DANN), figures | about 1.2 h |
| **Total** | **about 22 h** |

Decisions and fixes (full log: `outputs_v2/RUN_STATE.md`):
- **outputs_dir moved to `D:\shield_run\outputs_v2`.** v1's folder was not touched. The processed data in `D:\shield_run\data` was reused, because v2 changes no ingest or preprocessing code.
- **Pre-read the 49 M-row Bot-IoT (standard) training file into RAM** before its teacher ran. This avoided v1's 10-minute stall.
- **Lowered kd.yaml `jobs` from 3 to 2 at 15:47** (RAM exhausted, epochs 2-3x slower, GPU only 38% busy). I checked first that the 4.9 h KD tuning would not be redone. After the restart the GPU went back to 92%.
- **No experiment config was changed apart from that `jobs` line, and no code was changed.**

---

## 11. Files

All in `D:\shield_run\outputs_v2\`. Tables are in `paper/tables/` (CSV and LaTeX) and figures in `paper/figures/` (PNG and PDF).

| Item | Files |
|---|---|
| A, C Detection results | `tables/detection_<dataset>_<task>.csv`, `figures/pareto_*.png`, `figures/confusion_*.png` |
| B Tuning | `tables/kd_tuning_objective.csv`, `tables/kd_tuning_choice.csv`, `tuning/kd_choice.json` |
| C Feature budget | `tables/k_sweep_*.csv`, `figures/f1_vs_k_*.png` |
| C CD diagram (KD) | `figures/cd_kd_ablation.png`, `tables/kd_ablation_ranks.json` |
| D Low-data | `tables/lowdata_wilcoxon.csv`, `tables/lowdata_summary.csv` |
| E Cross-dataset | `tables/cross_zero_shot.csv`, `tables/cross_fewshot.csv`, `tables/cross_unseen_attacks.csv`, `cross/shap_agreement.json` |
| F Placement | `tables/placement_tuning.csv`, `tables/placement_ablation.csv`, `tables/placement_gap.csv`, `tables/placement_wilcoxon.csv`, `figures/cd_placement.png` |
| G Coupling | `tables/coupled_min_controllers.csv`, `figures/coupled_min_controllers.png` |
| Leakage (unchanged protocol) | `tables/leakage_audit_*.csv`, `tables/leakage_inflation.csv` |

The analysis scripts behind this report are in `final_analysis/` (`v2_analyse.py`, `v2_results.json`).

## 12. Decisions left for you

1. **The SHAP attribution loss** still shows no benefit in either version, or at low data. Present SHIELD as "SHAP-selected KD", or drop the loss claim.
2. **The hybrid's ants** contribute nothing measurable. Either present the method as "EHO plus memetic local search", or keep the ACO part and say so honestly.
3. **int8** is unreliable for the larger students. Consider leaving it out, or reporting it only for the binary tasks.
4. **Speed vs accuracy:** v2's bigger students halve the throughput on hard tasks. Decide which trade-off the paper should headline (v1 small/fast or v2 accurate).
5. **The borderline hybrid test** (28 of 30): relax the threshold, or keep it and accept it fails on this machine.
