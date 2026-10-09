# SHIELD-IoT v3: final results report

**Runs:** 7–9 October 2026, same desktop as v1/v2 (RTX 3060 12 GB, 20 threads, 32 GB RAM).
**Every table in this report is generated from the experiment output files** by `scripts/21_v3_report.py`
(the template is `results/v3/REPORT_TEMPLATE.md`; nothing is copied by hand). Table CSVs are in `results/v3/tables/`.
Numbers are **test macro-F1 in %** unless a column says otherwise.

## 0. Protocol (unchanged from v2, and applied to everything below)

- **Leakage controls:** `IAT` (CIC datasets) and `seq` (Bot-IoT) removed; exact duplicate feature vectors removed
  across train → validation → test; scaler fitted on train only.
- **Every choice is made on validation data**: thresholds, biases, ensemble weights, calibration, PTQ vs QAT,
  placement settings (on separate tuning graphs). **Test data is scored once**, after all choices are fixed.
- Paired tests are two-sided Wilcoxon signed-rank tests unless stated; "pairs" are task × seed (detection) or
  instances / topologies (placement).
- v1/v2 outputs were only read. v3 outputs: `D:\shield_run\outputs_v3`, full-data runs: `D:\shield_run\outputs_full`.

## 1. Summary of findings

| # | Experiment | Result | Strength of evidence |
|---|---|---|---|
| 1 | Full CICIoT2023 (6.4× more training data after de-dup), **identical test rows** | Same model (ensemble + rule): 34-class 70.98 → **73.22**, 8-class 74.60 → **77.01**. Best subset model of any kind on 34-class: 71.25 | Direct, row-matched; one GBDT seed in the ensemble |
| 2 | Validation-tuned per-class decision bias | Teacher +0.98 pts (35/40 better, p = 1.6e-9); SHIELD +0.67 (26/40, p = 1.5e-4); mixed on CICIoMT (students slightly worse on binary and 6-class) | Significant, small |
| 3 | Integer students (calibration chosen on validation, QAT when needed) | Mean int8 loss 14.93 → **1.05** pts; SHIELD int8 models 3.7–16.4 KB | Strong |
| 4 | Student → expert cascade | CICIoT 34-class: SHIELD 65.71 → 70.29 with 10% of flows escalated (ensemble expert 71.27) | Strong; thresholds set on validation |
| 5 | Cascade drives controller capacity | Same controller count as the student alone, accuracy close to the teacher | Model-based (M/M/1) |
| 6 | SHAP attribution loss | **More faithful explanations** (33/40, p = 2.3e-4), **no accuracy gain** | Significant; one task reversed |
| 7 | Stronger teacher (ResMLP×5 + XGBoost) | Teacher better on 7/8 tasks, **students distilled from it are not better** (all p ≥ 0.12) | Negative result |
| 8 | Placement: 18 methods, exact optima on 37 instances | No method dominates. GWO+SA has the best topology-level rank; the diversity-controlled hybrid has the best instance-level rank; the original EHO–ACO converges prematurely | Mixed; see §7 |
| 9 | Attack-surge isolation | Isolation removes attack exposure (0%) but costs benign throughput except at the largest surge | Model-based |
| 10 | Unseen-attack detection | Student catches 44–49% of unseen Web/BruteForce at 1% FPR; Mahalanobis/FlyHash novelty scores **do not help** | Negative result |

## 2. How much is left to gain? (identifiability)

1-NN error on validation vs the training set (Cover & Hart bounds on the Bayes error). Exact duplicate vectors with
conflicting labels are rare (about 0.004% of test rows, all DoS ↔ DDoS), so the exact ceiling is uninformative; the
neighbourhood estimate is not.

| task | 1-NN error (val) % | Bayes error bound % (Cover-Hart) | XGBoost test error % | Teacher s0 test error % |
|---|---|---|---|---|
| CICIoT 34-class | 17.24 | 9.04 - 17.24 | 11.83 | 13.38 |
| CICIoT 8-class | 14.13 | 7.38 - 14.13 | 9.87 | 11.34 |
| CICIoMT 19-class | 19.79 | 10.48 - 19.79 | 19.86 | 26.6 |
| CICIoMT 6-class | 19.37 | 10.32 - 19.37 | 18.57 | 19.86 |

**Reading:** on CICIoT, XGBoost's error (11.8%) is within about 3 points of the lower bound (9.0%). The largest
overlaps are DoS vs DDoS of the same protocol (for example DoS-UDP ↔ DDoS-UDP) and the rare web/brute-force classes
(1-NN F1 0–21%). These bounds are asymptotic estimates, not exact limits. CICIoMT's test set comes from different
captures, so its test error is not directly comparable with the validation-based bound.

## 3. More training data: full CICIoT2023 (the largest real gain)

**Design (directly comparable):** validation and test are **exactly** the strict CICIoT rows used in v1–v3
(re-derived with the same de-dup code and verified label-for-label by `scripts/20_verify_full_split.py`). Training
uses all 169 merged CSVs (46.7M rows) after removing exact duplicates (18.9M), label-conflicting vectors (1.0M) and
**every vector that occurs in the validation or test files (1.9M)**, leaving 25.8M rows (6.4× the subset). Student
settings (k, size) are reused from the subset's validation tuning.

Learning curve measured beforehand (XGBoost, validation, subset fractions):

| train rows | fraction | macro-F1 (val) | 8 rare classes | other 26 |
|---|---|---|---|---|
| 504279 | 0.125 | 64.78 | 7.91 | 82.28 |
| 1008561 | 0.25 | 66.27 | 11.1 | 83.25 |
| 2017118 | 0.5 | 67.58 | 14.74 | 83.84 |
| 4034235 | 1 | 69.03 | 18.13 | 84.69 |

The power-law fit predicted +3.9 macro-F1 for 9× data. The full data gives 6.4× after de-dup; the measured test
gain is below:

| task | teacher subset | teacher full | XGBoost subset | XGBoost full | ens4 subset | ens4 full | ens4+rule subset | ens4+rule full | SHIELD+rule subset | SHIELD+rule full | cascade 10% full | cascade 20% full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIoT binary | 94.15 | 94.74 | 95.11 | 95.57 | 95.41 | 95.7 | 95.41 | 95.7 | 94.27 | 94.54 | 95.7 | 95.7 |
| CICIoT 8-class | 71.15 | 74.09 | 72.93 | 75.07 | 74.23 | 77.11 | 74.6 | 77.01 | 69.49 | 69.6 | 75.81 | 76.36 |
| CICIoT 34-class | 68.09 | 69.39 | 68.56 | 70.81 | 70.76 | 72.63 | 70.98 | 73.22 | 65.65 | 67.47 | 72.13 | 72.68 |

- **34-class: 73.22** (ensemble + rule) on the identical test rows where the same model trained on the subset gave
  70.98 (and the best subset model of any kind, step 1's ensemble + rule, gave 71.25). XGBoost alone: 68.56 → 70.81.
- **The tiny student gains less** (SHIELD + rule 65.65 → 67.47 on 34-class; 69.49 → 69.60 on 8-class): it is limited
  by capacity more than data. The cascade (SHIELD → ensemble) reaches 72.68 at 20% escalation.
- Limitation: CICIoT2023's merged files have no capture identifiers, so a capture-level split (as in CICIoMT2024) is
  impossible. Exact duplicates are removed; near-duplicates within a capture cannot be. This applies to every
  published CICIoT2023 result too.

### Comparison with published results (indicative only)

| Task | **This work, strict, full data** | Published | Protocol of the published number |
|---|---|---|---|
| CICIoT 34-class | **73.22** | 71.4 (RF, Neto et al., Sensors 2023) | keeps `IAT`, no cross-split de-dup |
| | | 71.1 (LightGBM, H. M. Kim, manuscript under review 2026) | `IAT` removed; own split |
| CICIoT 8-class | **77.01** | 71.9 (RF, Neto et al. 2023) | keeps `IAT` |
| CICIoT binary | 95.70 | 96.5 (RF, Neto et al. 2023) | keeps `IAT` — **we do not beat this** |
| CICIoMT 19-class | 71.60 (ensemble) | 52.2–55.1 (RF, Dadkhah et al., IoT 2024; two secondary sources disagree) | dataset paper baseline; later papers report much higher scores under other protocols |

These published numbers use **different test sets** (and mostly a leaky feature), so the comparison is indicative;
only the subset → full comparison above is row-matched. We do **not** claim state of the art on CICIoMT2024: later
studies report far higher scores under different protocols, and a strict CICIoMT2024 reference was not found.

## 4. Better models on the original data (smaller gains)

### 4.1 Decision rules fitted on validation (logit adjustment + per-class bias)

| model | rule | pairs | mean gain (pts) | better | worse | p (Wilcoxon, two-sided) |
|---|---|---|---|---|---|---|
| kd_all | logit adjustment | 40 | -0 | 6 | 2 | 0.31 |
| kd_all | LA + per-class bias | 40 | 0.48 | 26 | 13 | 0.0012 |
| kd_mi | logit adjustment | 40 | 0.22 | 9 | 0 | 0.021 |
| kd_mi | LA + per-class bias | 40 | 0.77 | 25 | 9 | 0.00034 |
| kd_random | logit adjustment | 40 | -0.06 | 9 | 4 | 0.3 |
| kd_random | LA + per-class bias | 40 | 1.12 | 26 | 10 | 0.0034 |
| kd_shap | logit adjustment | 40 | 0.16 | 10 | 0 | 0.012 |
| kd_shap | LA + per-class bias | 40 | 0.55 | 24 | 12 | 0.005 |
| kd_variance | logit adjustment | 40 | 0.14 | 6 | 1 | 0.17 |
| kd_variance | LA + per-class bias | 40 | 2.66 | 27 | 7 | 2.3e-05 |
| scratch_all | logit adjustment | 40 | 0.14 | 13 | 1 | 0.0068 |
| scratch_all | LA + per-class bias | 40 | 0.58 | 28 | 7 | 0.00014 |
| shield | logit adjustment | 40 | 0.1 | 10 | 0 | 0.012 |
| shield | LA + per-class bias | 40 | 0.67 | 26 | 9 | 0.00015 |
| teacher | logit adjustment | 40 | 0 | 0 | 0 | 1 |
| teacher | LA + per-class bias | 40 | 0.98 | 35 | 5 | 1.6e-09 |
| v3-kd_all_ens | logit adjustment | 40 | 0.11 | 12 | 2 | 0.016 |
| v3-kd_all_ens | LA + per-class bias | 40 | 0.34 | 26 | 8 | 0.0015 |
| v3-kd_shap_ens | logit adjustment | 40 | 0.18 | 10 | 1 | 0.023 |
| v3-kd_shap_ens | LA + per-class bias | 40 | 0.62 | 23 | 10 | 0.00098 |
| xgboost | logit adjustment | 8 | 0.1 | 2 | 2 | 0.75 |
| xgboost | LA + per-class bias | 8 | 0.47 | 5 | 2 | 0.17 |

Per task (mean over seeds):

| tk | kd_all raw | shield raw | teacher raw | xgboost raw | kd_all +rule | shield +rule | teacher +rule | xgboost +rule |
|---|---|---|---|---|---|---|---|---|
| CICIoT binary | 94.72 | 94.28 | 94.15 | 95.11 | 94.74 | 94.27 | 95.11 | 95.11 |
| CICIoT 8-class | 69.35 | 68.78 | 71.15 | 72.93 | 69.91 | 69.49 | 72.18 | 73.77 |
| CICIoT 34-class | 65.18 | 64.35 | 68.09 | 68.56 | 66.47 | 65.65 | 69.62 | 69.44 |
| CICIoMT binary | 97.9 | 96.61 | 97.71 | 96.71 | 97.79 | 96.47 | 97.6 | 96.69 |
| CICIoMT 6-class | 83.2 | 82.83 | 80.65 | 79.75 | 82.95 | 82.64 | 81.31 | 80.86 |
| CICIoMT 19-class | 70.44 | 69.24 | 69.87 | 70.39 | 70.48 | 69.71 | 70.42 | 70.31 |
| Bot-IoT binary | 97.24 | 96.07 | 96.9 | 98.99 | 99.01 | 98.32 | 99.07 | 98.99 |
| Bot-IoT 5-class | 92.74 | 91.46 | 92.26 | 95.63 | 93.28 | 92.46 | 93.26 | 96.64 |

Gains concentrate on CICIoT 8/34-class and Bot-IoT. CICIoMT is mixed: the teacher and the 19-class task gain
(SHIELD 69.24 → 69.71), while the students lose 0.1–0.25 points on binary and 6-class; its test captures differ from
the validation slice, so validation-fitted biases transfer less reliably.

### 4.2 Stronger teacher and models

| task | v2 teacher (5-seed mean) | XGBoost | deep ensemble (5 ResMLP) | w_ResMLP (val) | ensemble | ensemble + rule |
|---|---|---|---|---|---|---|
| CICIoT binary | 94.15 | 95.11 | 94.34 | 0.4 | 95.25 | 95.32 |
| CICIoT 8-class | 71.15 | 72.93 | 72.92 | 0.7 | 74.33 | 74.5 |
| CICIoT 34-class | 68.09 | 68.57 | 69.31 | 0.6 | 70.69 | 71.25 |
| CICIoMT binary | 97.71 | 96.71 | 97.89 | 0 | 96.71 | 96.69 |
| CICIoMT 6-class | 80.65 | 79.75 | 81.21 | 0.4 | 81.62 | 81.42 |
| CICIoMT 19-class | 69.87 | 70.39 | 70.44 | 0.5 | 71.6 | 71.1 |
| Bot-IoT binary | 96.9 | 98.99 | 97.09 | 0.3 | 99.03 | 99.07 |
| Bot-IoT 5-class | 92.26 | 95.63 | 93.24 | 0.4 | 95.81 | 96.33 |

Students distilled from this ensemble, compared with the same students distilled from the v2 teacher:

| comparison | decision | pairs | mean diff (pts) | better | p |
|---|---|---|---|---|---|
| kd_all_ens vs kd_all | raw | 40 | -0.09 | 24 | 0.56 |
| kd_all_ens vs kd_all | +rule | 40 | -0.24 | 20 | 0.64 |
| kd_shap_ens vs kd_shap | raw | 40 | -0.05 | 14 | 0.12 |
| kd_shap_ens vs kd_shap | +rule | 40 | 0.02 | 22 | 0.85 |
| kd_shap_ens vs shield | raw | 40 | 0.09 | 21 | 0.54 |
| kd_shap_ens vs shield | +rule | 40 | 0.04 | 24 | 0.21 |

**A stronger teacher does not make better students** here: the 3–15k-parameter students are capacity-limited.

Further detectors (LightGBM, CatBoost on GPU, ResMLP on quantile-transformed inputs with cross-entropy or Balanced
Softmax, and validation-driven ensemble selection, Caruana et al. 2004):

| task | teacher | xgboost | lgbm | catboost | qt_ce | qt_bs | ens4 | ens4+rule |
|---|---|---|---|---|---|---|---|---|
| Bot-IoT binary | 97.09 | 98.99 | 89.14 | 98.91 | 82.06 | 64.85 | 99.35 | 99.22 |
| Bot-IoT 5-class | 93.24 | 95.63 | 72.76 | 94.59 | 88.22 | 74.47 | 96.48 | 96.49 |
| CICIoMT 19-class | 70.44 | 70.39 | 66.35 | 60.56 | 67.95 | 64.21 | 71.6 | 71.1 |
| CICIoMT binary | 97.89 | 96.71 | 96.99 | 97.05 | 97.28 | 97.22 | 96.91 | 96.91 |
| CICIoMT 6-class | 81.21 | 79.75 | 83.87 | 84.58 | 78.68 | 74.32 | 83.04 | 83.93 |
| CICIoT binary | 94.34 | 95.11 | 95.28 | 95.22 | 93.63 | 91.76 | 95.41 | 95.41 |
| CICIoT 34-class | 69.31 | 68.57 | 61.58 | 69.13 | 66.22 | 64.34 | 70.76 | 70.98 |
| CICIoT 8-class | 72.92 | 72.93 | 68.58 | 73.07 | 70.61 | 64.74 | 74.23 | 74.6 |
| CICIoT(full) binary | 94.86 | 95.57 | 95.54 | 95.47 | 93.88 | 92.17 | 95.7 | 95.7 |
| CICIoT(full) class34 | 70.81 | 70.81 | 63.16 | 68.29 | 68.11 | 65.29 | 72.63 | 73.22 |
| CICIoT(full) family | 75.39 | 75.07 | 70.12 | 73.75 | 72.62 | 64.86 | 77.11 | 77.01 |

Model changes move results by about ±1 point; quantile inputs and Balanced Softmax did not help the neural model.
CatBoost is the best single model on CICIoMT 6-class (84.58), but selecting a model by its test score would be
cherry-picking, so the validation-selected ensemble is what we report.

## 5. Deployment

### 5.1 Integer students

v2 used PyTorch dynamic quantisation (one input scale per batch; heavy-tailed features destroy the small ones). v3
uses integer-only inference with a per-feature input scale, per-channel int8 weights and calibrated activations; the
calibration (proportional vs class-balanced rows, clip percentiles) is chosen on validation, and QAT is used when
PTQ loses more than 0.5 points on validation. W8A16 keeps int8 weights with 16-bit activations.

| task | v2 dynamic int8 drop (pts) | v3 W8A8 drop (pts) | v3 W8A16 drop (pts) | SHIELD fp32 bytes | SHIELD int8 bytes |
|---|---|---|---|---|---|
| CICIoT binary | 20.96 | 0.05 | 0.03 | 12936 | 3992 |
| CICIoT 8-class | 27.36 | 1.17 | 0.99 | 52000 | 14536 |
| CICIoT 34-class | 23.86 | 1.41 | 1.17 | 58760 | 16408 |
| CICIoMT binary | 7.52 | 0.2 | 0.14 | 11912 | 3720 |
| CICIoMT 6-class | 9.39 | 1.26 | 1.49 | 43288 | 12280 |
| CICIoMT 19-class | 15.56 | 1.64 | 2.28 | 54860 | 15328 |
| Bot-IoT binary | 0.23 | -0.13 | -0.01 | 11912 | 3720 |
| Bot-IoT 5-class | 14.52 | 2.82 | 2.06 | 40980 | 11680 |
| mean (all 160 models) | 14.93 | 1.05 | 1.02 |  |  |

Rare-class values sit in the feature tails, which is why a single fixed clipping percentile failed in development
and the calibration is selected per model.

### 5.2 Cascade: tiny student, uncertain flows escalated

The student's least-confident flows (max softmax probability; threshold set on validation for each budget) go to an
expert. SHIELD student with the step-2 rule; "matched" = smallest budget whose validation F1 reaches the expert's.

| task | expert | student alone | 5% budget | 10% budget | 20% budget | expert alone | matched: F1 | matched: test escalated % |
|---|---|---|---|---|---|---|---|---|
| Bot-IoT 5-class | ensemble | 92.53 | 93.61 | 94.3 | 95.22 | 96.33 | 95.49 | 25 |
| Bot-IoT 5-class | teacher | 92.53 | 93.23 | 93.49 | 93.72 | 94.05 | 93.74 | 20 |
| Bot-IoT binary | ensemble | 98.32 | 99.08 | 99.08 | 99.08 | 99.07 | 99.02 | 0.8 |
| Bot-IoT binary | teacher | 98.32 | 99.18 | 99.18 | 99.18 | 99.18 | 99.08 | 0.8 |
| CICIoMT 19-class | ensemble | 69.61 | 71.05 | 71.53 | 71.93 | 71.16 | 71.54 | 57 |
| CICIoMT 19-class | teacher | 69.61 | 70 | 70.15 | 70.28 | 70.24 | 70.3 | 24.6 |
| CICIoMT 6-class | ensemble | 82.53 | 82.97 | 82.92 | 82.68 | 81.42 | 81.48 | 55.6 |
| CICIoMT 6-class | teacher | 82.53 | 82.57 | 82.5 | 82.19 | 81.05 | 82.3 | 27.7 |
| CICIoMT binary | ensemble | 96.39 | 96.69 | 96.69 | 96.69 | 96.69 | 96.72 | 1.4 |
| CICIoMT binary | teacher | 96.39 | 97.7 | 97.7 | 97.7 | 97.7 | 96.39 | 0 |
| CICIoT 34-class | ensemble | 65.71 | 69.73 | 70.29 | 70.73 | 71.27 | 70.83 | 24 |
| CICIoT 34-class | teacher | 65.71 | 67.81 | 68.02 | 68.13 | 68.23 | 67.81 | 5 |
| CICIoT 8-class | ensemble | 69.53 | 73.34 | 73.77 | 74.18 | 74.5 | 74.18 | 20.2 |
| CICIoT 8-class | teacher | 69.53 | 70.72 | 70.83 | 70.91 | 70.93 | 70.56 | 2.9 |
| CICIoT binary | ensemble | 94.27 | 95.3 | 95.31 | 95.31 | 95.31 | 95.06 | 1 |
| CICIoT binary | teacher | 94.27 | 95.05 | 95.06 | 95.06 | 95.06 | 94.92 | 1 |

On CICIoT 34-class, 10% escalation lifts SHIELD from 65.71 to 70.29 (ensemble alone 71.27). On CICIoMT 6-class,
escalating hurts (82.53 → 81.48 matched), again because of its validation → test shift. Test escalation rates match
the validation budget closely on CICIoT and Bot-IoT; on CICIoMT they exceed it (for example 57.0% at a 50% budget).

### 5.3 Controller coupling with the cascade

Controllers run the student on every flow and the teacher on escalated flows (cost 1/μ_student + e/μ_teacher).
Minimum controllers meeting the relative SLA (CICIoT 34-class capacities):

| topology | rho | shield | cascade_2pct | cascade_5pct | cascade_10pct | cascade_20pct | teacher |
|---|---|---|---|---|---|---|---|
| Geant2012 | 0.5 | 3 | 3 | 3 | 3 | 3 | 4 |
| Geant2012 | 0.7 | 3 | 3 | 3 | 3 | 3 | 6 |
| Geant2012 | 0.9 | 3 | 3 | 3 | 3 | 3 | 7 |
| Cogentco | 0.5 | 6 | 6 | 6 | 6 | 6 | 6 |
| Cogentco | 0.7 | 6 | 6 | 6 | 6 | 6 | 6 |
| Cogentco | 0.9 | 6 | 6 | 6 | 6 | 6 | 8 |
| syn200 | 0.5 | 7 | 7 | 7 | 7 | 7 | 7 |
| syn200 | 0.7 | 7 | 7 | 7 | 7 | 7 | 7 |
| syn200 | 0.9 | 7 | 7 | 7 | 7 | 7 | 8 |

| model | test macro-F1 (CICIoT 34) | test escalated % |
|---|---|---|
| shield | 65.71 | 0 |
| cascade_2pct | 67.42 | 2 |
| cascade_5pct | 67.81 | 5 |
| cascade_10pct | 68.02 | 10.1 |
| cascade_20pct | 68.13 | 20 |
| teacher | 68.23 | 100 |

The cascade needs **the same number of controllers as the student alone** at every load, while its accuracy
approaches the teacher's (68.13 at 20% vs 68.23). This is a queueing-model result, not a testbed measurement.

## 6. Explanation faithfulness of the SHAP loss

Spearman correlation between the student's and the teacher's SHAP feature rankings, SHIELD (with the attribution
loss) vs kd_shap (same features and size, no loss); 5 seeds × 8 tasks:

| task | kd_shap | shield |
|---|---|---|
| Bot-IoT 5-class | 0.708 | 0.885 |
| Bot-IoT binary | 0.453 | 0.32 |
| CICIoMT 19-class | 0.885 | 0.9 |
| CICIoMT 6-class | 0.814 | 0.856 |
| CICIoMT binary | 0.122 | 0.276 |
| CICIoT 34-class | 0.799 | 0.885 |
| CICIoT 8-class | 0.803 | 0.89 |
| CICIoT binary | 0.144 | 0.664 |
| mean over tasks | 0.591 | 0.709 |

SHIELD is more faithful in **33 of 40** pairs (mean +0.119, p = 2.3e-4). Bot-IoT binary goes the other way. Accuracy
is unchanged (v2: SHIELD vs kd_shap, p = 0.10). **Claim: the attribution loss preserves the teacher's explanation
behaviour; it does not improve accuracy.**

## 7. Controller placement

18 methods, each tuned with the same effort on separate tuning graphs and run with exactly 20,000 objective
evaluations (cache hits counted), 30 seeds × 56 instances (8 topologies × k × ρ). Exact optima by parallel exhaustive
search (`brute_force_parallel`, verified identical to the serial search) on 37 instances (up to 10^8 subsets).

| algorithm | gap % (22 small, v2 set) | gap % (all 37 exact) | gap % (15 medium/large exact) | worst gap % | rank (56 instances) | rank (8 topologies) | budget fraction to final | v2 gap % (22 small) |
|---|---|---|---|---|---|---|---|---|
| gwo_sa | 0.037 | 0.304 | 0.697 | 6.07 | 4.92 | 2.56 | 0.292 |  |
| hybrid_linear | 0.043 | 0.414 | 0.959 | 5.89 | 4.85 | 3.06 | 0.26 |  |
| hybrid_fuzzy | 0.069 | 0.696 | 1.616 | 14.25 | 6.52 | 4.44 | 0.333 |  |
| hybrid_eho_aco | 0.458 | 1.652 | 3.402 | 22.45 | 5.65 | 5.31 | 0.167 | 0.458 |
| sa | 0.133 | 1.173 | 2.7 | 15.81 | 6.66 | 5.56 | 0.294 | 0.133 |
| hybrid_no_ants | 0.269 | 2.399 | 5.522 | 29.16 | 6.14 | 7.19 | 0.177 | 0.269 |
| de | 0.24 | 1.586 | 3.559 | 24.57 | 7.72 | 7.19 | 0.32 |  |
| hho | 0.224 | 1.684 | 3.825 | 23.12 | 7.69 | 7.44 | 0.185 |  |
| pso | 0.107 | 1.199 | 2.8 | 20.67 | 8.93 | 8.94 | 0.438 | 0.107 |
| gwo | 0.242 | 1.872 | 4.263 | 31.45 | 10.05 | 9.31 | 0.512 |  |
| ga | 0.674 | 2.599 | 5.422 | 30.66 | 9.92 | 10.81 | 0.212 | 0.674 |
| foa | 0.56 | 2.537 | 5.435 | 26.74 | 9.73 | 11.06 | 0.132 |  |
| aco | 0.339 | 2.145 | 4.793 | 34.51 | 10.44 | 11.25 | 0.364 | 0.339 |
| eho | 0.396 | 2.67 | 6.007 | 29.6 | 13.47 | 13.31 | 0.536 | 0.396 |
| hybrid_no_ls | 2.102 | 4.514 | 8.05 | 48.63 | 14.51 | 15.25 | 0.429 | 2.102 |
| random | 0.639 | 4.575 | 10.347 | 43.41 | 15.21 | 15.81 | 0.417 | 0.639 |
| woa | 1.802 | 5.949 | 12.03 | 52.97 | 14.07 | 16 | 0.084 |  |
| hybrid_swap_ls | 1.836 | 5.4 | 10.627 | 48.32 | 14.51 | 16.5 | 0.112 | 1.836 |

"budget fraction to final" = share of the 20,000 evaluations used before the run is within 0.1% of its final value
(low = early stagnation).

Paired comparisons (method means; 56 instances, and 8 topologies as independent units):

| A | B | instances A better | instances A worse | p (56 instances) | topologies A better (of 8) | p (8 topologies) |
|---|---|---|---|---|---|---|
| hybrid_eho_aco | gwo_sa | 16 | 17 | 0.32 | 2 | 0.078 |
| hybrid_linear | gwo_sa | 18 | 15 | 0.79 | 3 | 0.69 |
| hybrid_linear | hybrid_eho_aco | 17 | 17 | 0.39 | 5 | 0.078 |
| hybrid_fuzzy | gwo_sa | 2 | 34 | 2.5e-07 | 0 | 0.016 |
| hybrid_fuzzy | hybrid_eho_aco | 15 | 24 | 0.64 | 4 | 0.3 |
| hybrid_eho_aco | hybrid_no_ants | 16 | 13 | 0.5 | 4 | 0.38 |

Per-instance tests over the 30 seeds (Holm-corrected), original hybrid vs each method:

| vs | hybrid_eho_aco significantly better (instances of 56) | hybrid_eho_aco significantly worse |
|---|---|---|
| hybrid_linear | 9 | 7 |
| hybrid_fuzzy | 14 | 4 |
| hybrid_no_ants | 4 | 0 |
| hybrid_no_ls | 41 | 0 |
| hybrid_swap_ls | 38 | 0 |
| eho | 34 | 1 |
| aco | 31 | 2 |
| ga | 17 | 0 |
| pso | 21 | 1 |
| sa | 9 | 1 |
| gwo | 25 | 0 |
| gwo_sa | 9 | 8 |
| foa | 16 | 0 |
| woa | 37 | 0 |
| hho | 11 | 1 |
| de | 13 | 3 |
| random | 41 | 0 |

Tuning-graph scores (lower is better; ratio to the best result found on each tuning graph):

| algorithm | tuning score | default score |
|---|---|---|
| hybrid_linear | 1.0059 | 1.0396 |
| hybrid_eho_aco | 1.0078 | 1.0332 |
| gwo_sa | 1.0083 | 1.0458 |
| hybrid_fuzzy | 1.0138 | 1.0389 |
| hho | 1.0184 | 1.0441 |
| pso | 1.0277 | 1.0964 |
| gwo | 1.0284 | 1.0284 |
| ga | 1.0308 | 1.108 |
| de | 1.0317 | 1.0811 |
| sa | 1.0319 | 1.0486 |
| aco | 1.0489 | 1.084 |
| foa | 1.0555 | 1.0769 |
| eho | 1.081 | 1.1018 |
| woa | 1.0838 | 1.1388 |

**What the evidence supports:**
- **No single best optimiser.** GWO+SA has the best topology-level rank (2.56) and lowest mean gap (0.304%); the
  diversity-controlled hybrid (`hybrid_linear`) has the best instance-level rank (4.85) and the lowest worst-case gap
  (5.89%). Neither differs significantly from the other (18 vs 15 instances, p = 0.79).
- **Diagnosis of the original EHO–ACO:** premature convergence. It is within 0.1% of its final value after 16.7% of
  the budget; on Cogentco/syn200 the pheromone entropy falls to its MAX-MIN floor and every elephant ends with the
  same k nodes (`analysis/pheromone_diag.py`).
- **Diversity control** (pheromone smoothing + more ant-rebuilt elephants when stagnation and low clan diversity are
  detected; Stützle & Hoos 2000) cuts the mean gap 1.652% → 0.414% and the worst case 22.45% → 5.89%. Per instance
  it is **not uniformly better** (the original is significantly better on 9 instances, worse on 7): it removes the
  catastrophic failures rather than improving typical runs.
- **The fuzzy version of the same controller is worse** than GWO+SA (2 vs 34 instances, p = 2.5e-7) and is not
  adopted. Note: before the proper run, untuned probes of both controllers were made on two evaluation topologies;
  they selected nothing, and all reported numbers come from the tuning-graph → frozen → full-grid protocol.
- **ACO's own contribution is small:** full hybrid vs no-ants is not significant at the aggregate level (16 vs 13
  instances, p = 0.5); per instance over seeds it is significantly better on 4 and worse on 0 (large graphs).
- **A redesigned ACO** (marginal-gain heuristic + MMAS reset) was tried and stopped: worse than the original on the
  tuning graphs (1.0153 vs 1.0078) and on the first completed Kdl instances, and about 20 h for the full grid. Its
  partial runs remain in `runs.jsonl` and are excluded from all tables.
- **Not claimed:** that EHO–ACO (in any version) is the best placement method.

## 8. Attack surge: does detector-driven isolation protect benign switches?

A placement optimised for normal load (ρ = 0.7) faces a surge multiplying the traffic of 10% of switches by M.
Flow costs use the measured cascade escalation rates (benign 15.0%, attack 4.6%; attacks are mostly confident
floods). Strategies are evaluated on the true load; the detector's flow-level precision 99.7%, recall 99.4%,
FPR 6.7% (CICIoT test) decide which switches are flagged. 3 topologies × 20 trials per M.
All network metrics come from the M/M/1 queueing model, not packet-level emulation; "time to mitigate" = model MTTD
(about 0.9 ms) + measured re-optimisation time, without switch-migration time.

| M | strategy | benign within SLA % | benign flows dropped % | all flows dropped % | attack exposure % | max utilisation | controllers | time to mitigate s |
|---|---|---|---|---|---|---|---|---|
| 5 | S0_static | 98.36 | 1.34 | 1.55 | 100 | 0.99 | 5.33 |  |
| 5 | S1_replace_k | 98.8 | 0.3 | 0.28 | 100 | 0.95 | 5.33 | 0.79 |
| 5 | S1_replace_k+1 | 98.82 | 0 | 0 | 100 | 0.92 | 6.33 | 0.89 |
| 5 | S2_isolate_k+1 | 98.22 | 2.61 | 9.43 | 0 | 1.26 | 6.33 | 0.72 |
| 10 | S0_static | 96.67 | 3.55 | 4.12 | 100 | 1.09 | 5.33 |  |
| 10 | S1_replace_k | 98.88 | 3.06 | 3.19 | 98.7 | 1.02 | 5.33 | 0.79 |
| 10 | S1_replace_k+1 | 98.96 | 0.04 | 0.17 | 98.7 | 0.94 | 6.33 | 0.89 |
| 10 | S2_isolate_k+1 | 98 | 4.23 | 22.08 | 0 | 1.84 | 6.33 | 0.7 |
| 20 | S0_static | 95.84 | 18.69 | 24.27 | 100 | 1.79 | 5.33 |  |
| 20 | S1_replace_k | 98.94 | 20.1 | 22.76 | 100 | 1.52 | 5.33 | 0.44 |
| 20 | S1_replace_k+1 | 98.96 | 6.66 | 8.42 | 97.54 | 1.2 | 6.33 | 0.5 |
| 20 | S2_isolate_k+1 | 98.13 | 6.71 | 45.06 | 0 | 3.12 | 6.33 | 0.42 |

| isolation (S2) vs | mean benign-drop diff (pts) | S2 lower in | p |
|---|---|---|---|
| S1_replace_k+1 | 2.28 | 28/180 | 6.5e-14 |
| S1_replace_k | -3.3 | 73/180 | 0.015 |
| S0_static | -3.34 | 78/180 | 0.0011 |

**Reading:** isolation (one security controller for flagged switches) gives **0% attack exposure** but drops more
benign flows than re-placing with the same number of controllers (k + 1), except at the largest surge (×20: 6.71% vs
6.66%). It beats doing nothing and same-k re-placement on benign drops. This is a trade-off, not a dominance result.

## 9. Attacks never seen in training (leave one family out)

Binary student trained without family F; thresholds at 1% FPR on benign validation flows; 3 seeds.

| dataset | unseen | student detect % | student+flyhash detect % | student+mahalanobis detect % | student test FPR % | student+flyhash test FPR % | student+mahalanobis test FPR % |
|---|---|---|---|---|---|---|---|
| ciciomt | DDoS | 100 | 100 | 100 | 3.7 | 6.9 | 2.4 |
| ciciomt | DoS | 100 | 100 | 100 | 2 | 6.7 | 1.4 |
| ciciomt | MQTT | 98.1 | 98 | 98.1 | 3.8 | 6.8 | 2.2 |
| ciciomt | Recon | 89 | 90.3 | 89.4 | 1 | 6.6 | 1.4 |
| ciciomt | Spoofing | 67.9 | 61.2 | 61 | 2.7 | 6.6 | 1.1 |
| ciciot | BruteForce | 48.5 | 41.6 | 42.7 | 1.2 | 1.2 | 1.1 |
| ciciot | DDoS | 100 | 100 | 100 | 1.2 | 1.2 | 1.1 |
| ciciot | DoS | 100 | 100 | 100 | 1.1 | 1.2 | 1.1 |
| ciciot | Mirai | 99.9 | 100 | 100 | 1.2 | 1.2 | 1.1 |
| ciciot | Recon | 58.7 | 55.9 | 55.6 | 1.2 | 1.2 | 1.1 |
| ciciot | Spoofing | 15.4 | 16 | 18.1 | 1.1 | 1.2 | 1.1 |
| ciciot | Web | 43.5 | 39.5 | 40.6 | 1.2 | 1.2 | 1.1 |

The student alone catches 48.5% of unseen BruteForce and 43.5% of unseen Web attacks (CICIoT). Neither novelty score
(Mahalanobis on the hidden layer, Lee et al. 2018; FlyHash, Dasgupta et al. 2017/2018) improves detection at the
same FPR budget, and FlyHash's FPR does not transfer to CICIoMT's test captures (6.6–6.9% vs a 1% target).

## 10. Limitations

1. CICIoT2023 has no capture identifiers: near-duplicate windows of one capture can sit in train and test (exact
   duplicates are removed). Applies to all published CICIoT2023 results.
2. Literature comparisons use other test sets and mostly a leaky feature; only subset → full is row-matched.
3. Gradient-boosting members and the ensembles use one seed; teacher/student results use 5 seeds.
4. Student settings for the full data reuse the subset's validation tuning (no re-tuning).
5. CICIoMT2024's test captures differ from its validation slice; validation-fitted rules and escalation transfer
   poorly there.
6. Controller coupling and surge results come from a queueing model, not an SDN testbed.
7. Placement statistics depend on the unit: instances (56) are not independent; topology-level tests (8 units) have
   low power.
8. The Nemenyi/Holm results for placement are within one objective function (weights from v2); a weight-sensitivity
   study was not run.

## 11. Files

- Tables: `results/v3/tables/*.csv`; facts used in the text: `results/v3/data/facts.json`.
- Analysis scripts and their outputs: `results/v3/analysis/` (identifiability, learning curve, placement and
  pheromone diagnostics, int8 calibration diagnostics).
- v3 pipeline scripts: `scripts/10_posthoc.py` … `scripts/21_v3_report.py`; new modules `eval/posthoc.py`,
  `eval/cascade.py`, `eval/openset.py`, `eval/predictions.py`, `models/quant.py`, `placement/swarm.py`.
- Raw outputs (not in git, large): `D:\shield_run\outputs_v3`, `D:\shield_run\outputs_full`.
